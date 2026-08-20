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

"""Base transform classes for refactoring operations.

Defines the abstract interface for code transformations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from victor_coding.refactor.analyzer import BaseCodeAnalyzer
from victor_coding.refactor.protocol import (
    CodeEdit,
    RefactorPreview,
    RefactorRequest,
    RefactorResult,
    RefactorRisk,
    RefactorType,
    SourceLocation,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class CodeTransform(Protocol):
    """Protocol for code transformations."""

    @property
    def refactor_type(self) -> RefactorType:
        """The type of refactoring this transform handles."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """Languages this transform supports."""
        ...

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate if the transform can be applied."""
        ...

    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate a preview of the transform."""
        ...

    def apply(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorResult:
        """Apply the transform."""
        ...


class BaseTransform(ABC):
    """Abstract base class for code transformations."""

    @property
    @abstractmethod
    def refactor_type(self) -> RefactorType:
        """The type of refactoring this transform handles."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """Languages this transform supports."""
        return ["python"]

    @property
    def risk_level(self) -> RefactorRisk:
        """Default risk level for this transform."""
        return RefactorRisk.LOW

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate if the transform can be applied.

        Args:
            request: Refactor request
            source: Source code

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Basic validation
        if request.refactor_type != self.refactor_type:
            errors.append(f"Transform type mismatch: expected {self.refactor_type}")

        return len(errors) == 0, errors

    @abstractmethod
    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate a preview of the transform.

        Args:
            request: Refactor request
            sources: Dict mapping file paths to source code
            analyzer: Code analyzer instance

        Returns:
            RefactorPreview with proposed edits
        """
        ...

    def apply(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorResult:
        """Apply the transform.

        Default implementation generates preview and applies edits.

        Args:
            request: Refactor request
            sources: Dict mapping file paths to source code
            analyzer: Code analyzer instance

        Returns:
            RefactorResult with applied changes
        """
        import time

        start_time = time.time()

        # Get preview
        preview = self.preview(request, sources, analyzer)

        if not preview.is_valid:
            return RefactorResult(
                request=request,
                success=False,
                error_message="; ".join(preview.errors),
            )

        # Apply edits (in reverse order to preserve positions)
        modified_files = set()
        edits_applied = 0

        # Group edits by file
        edits_by_file: dict[Path, list[CodeEdit]] = {}
        for edit in preview.edits:
            file_path = edit.file_path
            if file_path not in edits_by_file:
                edits_by_file[file_path] = []
            edits_by_file[file_path].append(edit)

        # Apply edits to each file
        for file_path, edits in edits_by_file.items():
            if file_path not in sources:
                continue

            source = sources[file_path]
            lines = source.split("\n")

            # Sort edits in reverse order (by line, then column)
            edits.sort(
                key=lambda e: (e.location.start_line, e.location.start_column),
                reverse=True,
            )

            for edit in edits:
                lines = self._apply_edit(lines, edit)
                edits_applied += 1

            # Write back
            try:
                file_path.write_text("\n".join(lines))
                modified_files.add(file_path)
            except Exception as e:
                return RefactorResult(
                    request=request,
                    success=False,
                    error_message=f"Failed to write {file_path}: {e}",
                )

        duration_ms = (time.time() - start_time) * 1000

        return RefactorResult(
            request=request,
            success=True,
            edits_applied=edits_applied,
            files_modified=list(modified_files),
            duration_ms=duration_ms,
        )

    def _apply_edit(
        self,
        lines: list[str],
        edit: CodeEdit,
    ) -> list[str]:
        """Apply a single edit to lines.

        Args:
            lines: Source lines
            edit: Edit to apply

        Returns:
            Modified lines
        """
        loc = edit.location
        start_line = loc.start_line - 1  # 0-indexed
        end_line = loc.end_line - 1

        if start_line == end_line:
            # Single line edit
            line = lines[start_line]
            new_line = line[: loc.start_column] + edit.new_text + line[loc.end_column :]
            lines[start_line] = new_line
        else:
            # Multi-line edit
            first_line = lines[start_line][: loc.start_column]
            last_line = lines[end_line][loc.end_column :]

            new_lines = edit.new_text.split("\n")
            new_lines[0] = first_line + new_lines[0]
            new_lines[-1] = new_lines[-1] + last_line

            lines = lines[:start_line] + new_lines + lines[end_line + 1 :]

        return lines

    def _get_source_range(
        self,
        source: str,
        location: SourceLocation,
    ) -> str:
        """Extract source code for a location.

        Args:
            source: Full source code
            location: Location to extract

        Returns:
            Source code at location
        """
        lines = source.split("\n")
        start_line = location.start_line - 1
        end_line = location.end_line - 1

        if start_line == end_line:
            return lines[start_line][location.start_column : location.end_column]

        result = [lines[start_line][location.start_column :]]
        for i in range(start_line + 1, end_line):
            result.append(lines[i])
        result.append(lines[end_line][: location.end_column])

        return "\n".join(result)
