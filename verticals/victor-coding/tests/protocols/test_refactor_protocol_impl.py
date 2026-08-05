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

"""Tests for refactoring protocol types and data structures."""

import pytest
from pathlib import Path

pytest.importorskip("victor_coding")

from victor_coding.refactor.protocol import (
    CodeEdit,
    CodePattern,
    RefactorCapabilities,
    RefactorPreview,
    RefactorRequest,
    RefactorResult,
    RefactorRisk,
    RefactorScope,
    RefactorSuggestion,
    RefactorType,
    SourceLocation,
    Symbol,
)

# =============================================================================
# ENUM TESTS
# =============================================================================


class TestRefactorType:
    """Tests for RefactorType enum."""

    def test_rename_operations(self):
        """Test rename operation types."""
        assert RefactorType.RENAME_SYMBOL.value == "rename_symbol"
        assert RefactorType.RENAME_FILE.value == "rename_file"

    def test_extract_operations(self):
        """Test extract operation types."""
        assert RefactorType.EXTRACT_FUNCTION.value == "extract_function"
        assert RefactorType.EXTRACT_METHOD.value == "extract_method"
        assert RefactorType.EXTRACT_VARIABLE.value == "extract_variable"
        assert RefactorType.EXTRACT_CLASS.value == "extract_class"

    def test_inline_operations(self):
        """Test inline operation types."""
        assert RefactorType.INLINE_FUNCTION.value == "inline_function"
        assert RefactorType.INLINE_VARIABLE.value == "inline_variable"

    def test_move_operations(self):
        """Test move operation types."""
        assert RefactorType.MOVE_SYMBOL.value == "move_symbol"
        assert RefactorType.MOVE_TO_FILE.value == "move_to_file"

    def test_signature_operations(self):
        """Test signature change operations."""
        assert RefactorType.ADD_PARAMETER.value == "add_parameter"
        assert RefactorType.REMOVE_PARAMETER.value == "remove_parameter"
        assert RefactorType.REORDER_PARAMETERS.value == "reorder_parameters"

    def test_modernization_operations(self):
        """Test modernization operations."""
        assert RefactorType.CONVERT_TO_ASYNC.value == "convert_to_async"
        assert RefactorType.CONVERT_LOOP_TO_COMPREHENSION.value == "convert_loop_to_comprehension"


class TestRefactorScope:
    """Tests for RefactorScope enum."""

    def test_local_scope(self):
        """Test local scope."""
        assert RefactorScope.LOCAL.value == "local"

    def test_project_scope(self):
        """Test project scope."""
        assert RefactorScope.PROJECT.value == "project"

    def test_workspace_scope(self):
        """Test workspace scope."""
        assert RefactorScope.WORKSPACE.value == "workspace"


class TestRefactorRisk:
    """Tests for RefactorRisk enum."""

    def test_safe_risk(self):
        """Test safe risk level."""
        assert RefactorRisk.SAFE.value == "safe"

    def test_low_risk(self):
        """Test low risk level."""
        assert RefactorRisk.LOW.value == "low"

    def test_medium_risk(self):
        """Test medium risk level."""
        assert RefactorRisk.MEDIUM.value == "medium"

    def test_high_risk(self):
        """Test high risk level."""
        assert RefactorRisk.HIGH.value == "high"

    def test_breaking_risk(self):
        """Test breaking risk level."""
        assert RefactorRisk.BREAKING.value == "breaking"


# =============================================================================
# SOURCE LOCATION TESTS
# =============================================================================


class TestSourceLocation:
    """Tests for SourceLocation dataclass."""

    @pytest.fixture
    def sample_location(self):
        """Create sample source location."""
        return SourceLocation(
            file_path=Path("src/main.py"),
            start_line=10,
            start_column=5,
            end_line=15,
            end_column=20,
        )

    def test_creation(self, sample_location):
        """Test source location creation."""
        assert sample_location.file_path == Path("src/main.py")
        assert sample_location.start_line == 10
        assert sample_location.start_column == 5
        assert sample_location.end_line == 15
        assert sample_location.end_column == 20

    def test_contains_position_inside(self, sample_location):
        """Test contains with position inside."""
        assert sample_location.contains(12, 10) is True

    def test_contains_position_at_start(self, sample_location):
        """Test contains with position at start."""
        assert sample_location.contains(10, 5) is True

    def test_contains_position_at_end(self, sample_location):
        """Test contains with position at end."""
        assert sample_location.contains(15, 20) is True

    def test_contains_position_before_start_line(self, sample_location):
        """Test contains with position before start line."""
        assert sample_location.contains(5, 10) is False

    def test_contains_position_after_end_line(self, sample_location):
        """Test contains with position after end line."""
        assert sample_location.contains(20, 10) is False

    def test_contains_position_before_start_column(self, sample_location):
        """Test contains with position before start column on start line."""
        assert sample_location.contains(10, 3) is False

    def test_contains_position_after_end_column(self, sample_location):
        """Test contains with position after end column on end line."""
        assert sample_location.contains(15, 25) is False

    def test_contains_single_line_location(self):
        """Test contains for single-line location."""
        loc = SourceLocation(
            file_path=Path("test.py"),
            start_line=5,
            start_column=10,
            end_line=5,
            end_column=20,
        )
        assert loc.contains(5, 15) is True
        assert loc.contains(5, 5) is False
        assert loc.contains(5, 25) is False


# =============================================================================
# SYMBOL TESTS
# =============================================================================


class TestSymbol:
    """Tests for Symbol dataclass."""

    @pytest.fixture
    def sample_location(self):
        """Create sample source location."""
        return SourceLocation(
            file_path=Path("src/main.py"),
            start_line=10,
            start_column=0,
            end_line=20,
            end_column=0,
        )

    @pytest.fixture
    def sample_symbol(self, sample_location):
        """Create sample symbol."""
        return Symbol(
            name="my_function",
            kind="function",
            location=sample_location,
            scope="module.MyClass",
            type_annotation="int",
            docstring="A sample function",
            modifiers=["public", "async"],
        )

    def test_creation(self, sample_symbol):
        """Test symbol creation."""
        assert sample_symbol.name == "my_function"
        assert sample_symbol.kind == "function"
        assert sample_symbol.scope == "module.MyClass"

    def test_qualified_name_with_scope(self, sample_symbol):
        """Test qualified name with scope."""
        assert sample_symbol.qualified_name == "module.MyClass.my_function"

    def test_qualified_name_without_scope(self, sample_location):
        """Test qualified name without scope."""
        symbol = Symbol(name="global_func", kind="function", location=sample_location)
        assert symbol.qualified_name == "global_func"

    def test_modifiers(self, sample_symbol):
        """Test symbol modifiers."""
        assert "public" in sample_symbol.modifiers
        assert "async" in sample_symbol.modifiers

    def test_references_default(self, sample_location):
        """Test references default to empty list."""
        symbol = Symbol(name="test", kind="variable", location=sample_location)
        assert symbol.references == []


# =============================================================================
# CODE EDIT TESTS
# =============================================================================


class TestCodeEdit:
    """Tests for CodeEdit dataclass."""

    @pytest.fixture
    def sample_location(self):
        """Create sample source location."""
        return SourceLocation(
            file_path=Path("src/main.py"),
            start_line=10,
            start_column=0,
            end_line=10,
            end_column=20,
        )

    @pytest.fixture
    def sample_edit(self, sample_location):
        """Create sample code edit."""
        return CodeEdit(
            location=sample_location,
            new_text="def new_function():",
            description="Renamed function",
        )

    def test_creation(self, sample_edit):
        """Test code edit creation."""
        assert sample_edit.new_text == "def new_function():"
        assert sample_edit.description == "Renamed function"

    def test_file_path_property(self, sample_edit):
        """Test file_path property."""
        assert sample_edit.file_path == Path("src/main.py")


# =============================================================================
# REFACTOR REQUEST TESTS
# =============================================================================


class TestRefactorRequest:
    """Tests for RefactorRequest dataclass."""

    @pytest.fixture
    def sample_location(self):
        """Create sample source location."""
        return SourceLocation(
            file_path=Path("src/main.py"),
            start_line=10,
            start_column=0,
            end_line=10,
            end_column=20,
        )

    def test_creation_minimal(self, sample_location):
        """Test minimal request creation."""
        request = RefactorRequest(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=sample_location,
        )
        assert request.refactor_type == RefactorType.RENAME_SYMBOL
        assert request.scope == RefactorScope.PROJECT  # default

    def test_creation_with_rename(self, sample_location):
        """Test request with rename."""
        request = RefactorRequest(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=sample_location,
            new_name="new_function_name",
        )
        assert request.new_name == "new_function_name"

    def test_creation_with_extract(self, sample_location):
        """Test request with extract."""
        request = RefactorRequest(
            refactor_type=RefactorType.EXTRACT_FUNCTION,
            target=sample_location,
            extract_name="extracted_function",
        )
        assert request.extract_name == "extracted_function"

    def test_creation_with_move(self, sample_location):
        """Test request with move."""
        request = RefactorRequest(
            refactor_type=RefactorType.MOVE_TO_FILE,
            target=sample_location,
            destination_file=Path("src/utils.py"),
        )
        assert request.destination_file == Path("src/utils.py")

    def test_default_options(self, sample_location):
        """Test default options."""
        request = RefactorRequest(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=sample_location,
        )
        assert request.preview is True
        assert request.update_references is True
        assert request.update_imports is True
        assert request.preserve_formatting is True


# =============================================================================
# REFACTOR PREVIEW TESTS
# =============================================================================


class TestRefactorPreview:
    """Tests for RefactorPreview dataclass."""

    @pytest.fixture
    def sample_request(self):
        """Create sample request."""
        loc = SourceLocation(Path("test.py"), 1, 0, 1, 10)
        return RefactorRequest(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=loc,
        )

    def test_is_valid_no_errors(self, sample_request):
        """Test is_valid with no errors."""
        preview = RefactorPreview(
            request=sample_request,
            edits=[],
            errors=[],
        )
        assert preview.is_valid is True

    def test_is_valid_with_errors(self, sample_request):
        """Test is_valid with errors."""
        preview = RefactorPreview(
            request=sample_request,
            edits=[],
            errors=["Symbol not found"],
        )
        assert preview.is_valid is False

    def test_edit_count(self, sample_request):
        """Test edit_count property."""
        loc = SourceLocation(Path("test.py"), 1, 0, 1, 10)
        edits = [
            CodeEdit(loc, "new1"),
            CodeEdit(loc, "new2"),
            CodeEdit(loc, "new3"),
        ]
        preview = RefactorPreview(request=sample_request, edits=edits)
        assert preview.edit_count == 3

    def test_file_count(self, sample_request):
        """Test file_count property."""
        preview = RefactorPreview(
            request=sample_request,
            affected_files=[Path("a.py"), Path("b.py")],
        )
        assert preview.file_count == 2

    def test_default_risk(self, sample_request):
        """Test default risk level."""
        preview = RefactorPreview(request=sample_request)
        assert preview.risk == RefactorRisk.SAFE


# =============================================================================
# REFACTOR RESULT TESTS
# =============================================================================


class TestRefactorResult:
    """Tests for RefactorResult dataclass."""

    @pytest.fixture
    def sample_request(self):
        """Create sample request."""
        loc = SourceLocation(Path("test.py"), 1, 0, 1, 10)
        return RefactorRequest(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=loc,
        )

    def test_successful_result(self, sample_request):
        """Test successful refactor result."""
        result = RefactorResult(
            request=sample_request,
            success=True,
            edits_applied=5,
            files_modified=[Path("a.py"), Path("b.py")],
        )
        assert result.success is True
        assert result.edits_applied == 5
        assert len(result.files_modified) == 2

    def test_failed_result(self, sample_request):
        """Test failed refactor result."""
        result = RefactorResult(
            request=sample_request,
            success=False,
            error_message="Symbol not found",
        )
        assert result.success is False
        assert result.error_message == "Symbol not found"

    def test_can_undo_with_backups(self, sample_request):
        """Test can_undo with backups."""
        result = RefactorResult(
            request=sample_request,
            success=True,
            backup_paths=[Path("backup/a.py.bak")],
        )
        assert result.can_undo() is True

    def test_can_undo_without_backups(self, sample_request):
        """Test can_undo without backups."""
        result = RefactorResult(
            request=sample_request,
            success=True,
            backup_paths=[],
        )
        assert result.can_undo() is False


# =============================================================================
# REFACTOR CAPABILITIES TESTS
# =============================================================================


class TestRefactorCapabilities:
    """Tests for RefactorCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capabilities."""
        caps = RefactorCapabilities()
        assert caps.supported_refactors == []
        assert caps.supported_languages == []
        assert caps.supports_preview is True
        assert caps.supports_undo is True
        assert caps.supports_cross_file is True
        assert caps.max_affected_files == 100

    def test_custom_capabilities(self):
        """Test custom capabilities."""
        caps = RefactorCapabilities(
            supported_refactors=[
                RefactorType.RENAME_SYMBOL,
                RefactorType.EXTRACT_FUNCTION,
            ],
            supported_languages=["python", "typescript"],
            supports_preview=True,
            supports_undo=False,
            max_affected_files=50,
        )
        assert len(caps.supported_refactors) == 2
        assert "python" in caps.supported_languages
        assert caps.supports_undo is False
        assert caps.max_affected_files == 50


# =============================================================================
# CODE PATTERN TESTS
# =============================================================================


class TestCodePattern:
    """Tests for CodePattern dataclass."""

    def test_creation_minimal(self):
        """Test minimal pattern creation."""
        pattern = CodePattern(name="unused_import", pattern="import .*")
        assert pattern.name == "unused_import"
        assert pattern.pattern == "import .*"
        assert pattern.pattern_type == "ast"  # default

    def test_creation_full(self):
        """Test full pattern creation."""
        pattern = CodePattern(
            name="deprecated_call",
            pattern="deprecated_func\\(.*\\)",
            pattern_type="regex",
            language="python",
            description="Calls to deprecated function",
            replacement="new_func($1)",
        )
        assert pattern.language == "python"
        assert pattern.replacement == "new_func($1)"


# =============================================================================
# REFACTOR SUGGESTION TESTS
# =============================================================================


class TestRefactorSuggestion:
    """Tests for RefactorSuggestion dataclass."""

    @pytest.fixture
    def sample_location(self):
        """Create sample source location."""
        return SourceLocation(Path("test.py"), 10, 0, 20, 0)

    def test_creation(self, sample_location):
        """Test suggestion creation."""
        suggestion = RefactorSuggestion(
            refactor_type=RefactorType.EXTRACT_FUNCTION,
            target=sample_location,
            reason="Function is too long",
            confidence=0.85,
            suggested_name="extract_helper",
        )
        assert suggestion.refactor_type == RefactorType.EXTRACT_FUNCTION
        assert suggestion.reason == "Function is too long"
        assert suggestion.confidence == 0.85
        assert suggestion.suggested_name == "extract_helper"

    def test_defaults(self, sample_location):
        """Test suggestion defaults."""
        suggestion = RefactorSuggestion(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=sample_location,
            reason="Poor naming",
        )
        assert suggestion.confidence == 0.0
        assert suggestion.risk == RefactorRisk.SAFE
        assert suggestion.auto_fixable is False
        assert suggestion.suggested_name is None

    def test_high_confidence_auto_fixable(self, sample_location):
        """Test high confidence auto-fixable suggestion."""
        suggestion = RefactorSuggestion(
            refactor_type=RefactorType.ORGANIZE_IMPORTS,
            target=sample_location,
            reason="Imports not organized",
            confidence=1.0,
            risk=RefactorRisk.SAFE,
            auto_fixable=True,
        )
        assert suggestion.confidence == 1.0
        assert suggestion.auto_fixable is True
