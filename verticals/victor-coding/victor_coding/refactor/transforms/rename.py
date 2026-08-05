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

"""Rename refactoring transforms."""

import re
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


class RenameSymbolTransform(BaseTransform):
    """Transform for renaming symbols (variables, functions, classes)."""

    @property
    def refactor_type(self) -> RefactorType:
        return RefactorType.RENAME_SYMBOL

    @property
    def risk_level(self) -> RefactorRisk:
        return RefactorRisk.LOW

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate rename request."""
        is_valid, errors = super().validate(request, source)

        if not request.new_name:
            errors.append("New name is required for rename")
            is_valid = False

        if request.new_name:
            # Check if new name is valid identifier
            if not request.new_name.isidentifier():
                errors.append(f"'{request.new_name}' is not a valid identifier")
                is_valid = False

            # Check for Python keywords
            import keyword

            if keyword.iskeyword(request.new_name):
                errors.append(f"'{request.new_name}' is a Python keyword")
                is_valid = False

        return is_valid, errors

    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate rename preview."""
        preview = RefactorPreview(
            request=request,
            risk=self.risk_level,
        )

        # Validate
        target_file = request.target.file_path
        if target_file not in sources:
            preview.errors.append(f"File not found: {target_file}")
            return preview

        source = sources[target_file]
        is_valid, errors = self.validate(request, source)
        if not is_valid:
            preview.errors.extend(errors)
            return preview

        # Find symbol at target location
        symbol = analyzer.find_symbol_at(
            source,
            target_file,
            request.target.start_line,
            request.target.start_column,
        )

        if symbol is None:
            preview.errors.append("No symbol found at target location")
            return preview

        old_name = symbol.name
        new_name = request.new_name

        # Check if new name already exists
        all_symbols = analyzer.find_all_symbols(source, target_file)
        for s in all_symbols:
            if s.name == new_name and s.scope == symbol.scope:
                preview.errors.append(
                    f"Symbol '{new_name}' already exists in scope '{symbol.scope}'"
                )
                return preview

        # Find all references across files
        for file_path, file_source in sources.items():
            references = analyzer.find_references(file_source, file_path, old_name)

            for ref in references:
                edit = CodeEdit(
                    location=ref,
                    new_text=new_name,
                    description=f"Rename '{old_name}' to '{new_name}'",
                )
                preview.edits.append(edit)

            if references:
                preview.affected_files.append(file_path)

        if not preview.edits:
            preview.warnings.append("No references found to rename")

        return preview


class RenameFileTransform(BaseTransform):
    """Transform for renaming files and updating imports."""

    @property
    def refactor_type(self) -> RefactorType:
        return RefactorType.RENAME_FILE

    @property
    def risk_level(self) -> RefactorRisk:
        return RefactorRisk.MEDIUM

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate file rename request."""
        is_valid, errors = super().validate(request, source)

        if not request.destination_file:
            errors.append("Destination file path is required")
            is_valid = False

        if request.destination_file and request.destination_file.exists():
            errors.append(f"Destination file already exists: {request.destination_file}")
            is_valid = False

        return is_valid, errors

    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate file rename preview."""
        preview = RefactorPreview(
            request=request,
            risk=self.risk_level,
        )

        source_file = request.target.file_path
        dest_file = request.destination_file

        if source_file not in sources:
            preview.errors.append(f"Source file not found: {source_file}")
            return preview

        is_valid, errors = self.validate(request, sources[source_file])
        if not is_valid:
            preview.errors.extend(errors)
            return preview

        # Calculate old and new module names
        old_module = source_file.stem
        new_module = dest_file.stem

        # Find imports to update across all files
        for file_path, source in sources.items():
            if file_path == source_file:
                continue

            # Find imports of the old module
            import_pattern = rf"\bimport\s+{re.escape(old_module)}\b"
            from_pattern = rf"\bfrom\s+{re.escape(old_module)}\b"

            lines = source.split("\n")
            for i, line in enumerate(lines):
                if re.search(import_pattern, line) or re.search(from_pattern, line):
                    new_line = re.sub(
                        rf"\b{re.escape(old_module)}\b",
                        new_module,
                        line,
                    )
                    edit = CodeEdit(
                        location=SourceLocation(
                            file_path=file_path,
                            start_line=i + 1,
                            start_column=0,
                            end_line=i + 1,
                            end_column=len(line),
                        ),
                        new_text=new_line,
                        # human-readable preview text, not SQL
                        description=f"Update import from '{old_module}' to '{new_module}'",  # nosemgrep: python.django.security.injection.tainted-sql-string.tainted-sql-string
                    )
                    preview.edits.append(edit)

                    if file_path not in preview.affected_files:
                        preview.affected_files.append(file_path)

        # Add note about file rename (actual rename handled separately)
        preview.warnings.append(f"File will be renamed from '{source_file}' to '{dest_file}'")
        preview.affected_files.append(source_file)

        return preview
