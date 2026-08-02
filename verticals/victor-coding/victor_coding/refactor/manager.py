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

"""Refactoring manager for orchestrating refactoring operations.

Provides a high-level API for code refactoring.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from victor_coding.refactor.analyzer import BaseCodeAnalyzer, get_analyzer
from victor_coding.refactor.protocol import (
    RefactorCapabilities,
    RefactorPreview,
    RefactorRequest,
    RefactorResult,
    RefactorScope,
    RefactorSuggestion,
    RefactorType,
    SourceLocation,
)
from victor_coding.refactor.transforms.base import BaseTransform
from victor_coding.refactor.transforms.extract import (
    ExtractConstantTransform,
    ExtractFunctionTransform,
    ExtractVariableTransform,
)
from victor_coding.refactor.transforms.rename import (
    RenameFileTransform,
    RenameSymbolTransform,
)

logger = logging.getLogger(__name__)


class RefactorManager:
    """High-level manager for refactoring operations.

    Orchestrates code analysis, transform selection, and
    refactoring application.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
    ):
        """Initialize the refactor manager.

        Args:
            project_root: Root directory of the project
            backup_dir: Directory for backup files
        """
        self.project_root = project_root or Path.cwd()
        self.backup_dir = backup_dir or (self.project_root / ".refactor_backups")

        # Initialize transforms
        self._transforms: dict[RefactorType, BaseTransform] = {}
        self._register_builtin_transforms()

        # Initialize analyzers
        self._analyzers: dict[str, BaseCodeAnalyzer] = {}

    def _register_builtin_transforms(self) -> None:
        """Register built-in transforms."""
        transforms = [
            RenameSymbolTransform(),
            RenameFileTransform(),
            ExtractFunctionTransform(),
            ExtractVariableTransform(),
            ExtractConstantTransform(),
        ]
        for transform in transforms:
            self._transforms[transform.refactor_type] = transform

    def register_transform(self, transform: BaseTransform) -> None:
        """Register a custom transform.

        Args:
            transform: Transform to register
        """
        self._transforms[transform.refactor_type] = transform

    def get_capabilities(self) -> RefactorCapabilities:
        """Get refactoring capabilities.

        Returns:
            RefactorCapabilities describing available operations
        """
        return RefactorCapabilities(
            supported_refactors=list(self._transforms.keys()),
            supported_languages=["python"],  # Start with Python
            supports_preview=True,
            supports_undo=True,
            supports_cross_file=True,
        )

    def get_analyzer(self, language: str) -> Optional[BaseCodeAnalyzer]:
        """Get or create an analyzer for a language.

        Args:
            language: Language identifier

        Returns:
            Analyzer instance or None
        """
        if language not in self._analyzers:
            analyzer = get_analyzer(language)
            if analyzer:
                self._analyzers[language] = analyzer
        return self._analyzers.get(language)

    def preview(
        self,
        refactor_type: RefactorType,
        target_file: Path,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        scope: RefactorScope = RefactorScope.PROJECT,
        **kwargs,
    ) -> RefactorPreview:
        """Generate a preview of a refactoring operation.

        Args:
            refactor_type: Type of refactoring
            target_file: File containing target code
            start_line: Start line of target
            start_column: Start column
            end_line: End line
            end_column: End column
            scope: Scope of refactoring
            **kwargs: Additional parameters (new_name, extract_name, etc.)

        Returns:
            RefactorPreview with proposed changes
        """
        # Build request
        request = RefactorRequest(
            refactor_type=refactor_type,
            target=SourceLocation(
                file_path=target_file,
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
            ),
            scope=scope,
            preview=True,
            new_name=kwargs.get("new_name"),
            extract_name=kwargs.get("extract_name"),
            destination_file=kwargs.get("destination_file"),
        )

        return self.preview_request(request)

    def preview_request(self, request: RefactorRequest) -> RefactorPreview:
        """Generate preview from a RefactorRequest.

        Args:
            request: Refactor request

        Returns:
            RefactorPreview
        """
        preview = RefactorPreview(request=request)

        # Get transform
        transform = self._transforms.get(request.refactor_type)
        if transform is None:
            preview.errors.append(f"No transform available for {request.refactor_type}")
            return preview

        # Detect language
        language = self._detect_language(request.target.file_path)
        analyzer = self.get_analyzer(language)
        if analyzer is None:
            preview.errors.append(f"No analyzer available for language: {language}")
            return preview

        # Load sources based on scope
        sources = self._load_sources(request.target.file_path, request.scope)
        if not sources:
            preview.errors.append("Failed to load source files")
            return preview

        # Generate preview
        return transform.preview(request, sources, analyzer)

    def apply(
        self,
        request: RefactorRequest,
        create_backup: bool = True,
    ) -> RefactorResult:
        """Apply a refactoring operation.

        Args:
            request: Refactor request
            create_backup: Whether to create backups

        Returns:
            RefactorResult
        """
        # Get preview first
        preview = self.preview_request(request)

        if not preview.is_valid:
            return RefactorResult(
                request=request,
                success=False,
                error_message="; ".join(preview.errors),
            )

        # Create backups
        backup_paths = []
        if create_backup:
            backup_paths = self._create_backups(preview.affected_files)

        # Get transform and analyzer
        transform = self._transforms[request.refactor_type]
        language = self._detect_language(request.target.file_path)
        analyzer = self.get_analyzer(language)

        # Load sources
        sources = self._load_sources(request.target.file_path, request.scope)

        # Apply transform
        request.preview = False  # Actually apply changes
        result = transform.apply(request, sources, analyzer)
        result.backup_paths = backup_paths

        return result

    def undo(self, result: RefactorResult) -> bool:
        """Undo a refactoring operation.

        Args:
            result: Result to undo

        Returns:
            True if undo succeeded
        """
        if not result.can_undo():
            logger.warning("Cannot undo: no backups available")
            return False

        try:
            for backup_path in result.backup_paths:
                # Extract original path from backup name
                # Backup format: original_name.backup_timestamp
                original_name = backup_path.stem.rsplit(".backup_", 1)[0]

                # Find the original file
                for modified in result.files_modified:
                    if modified.stem == original_name:
                        shutil.copy(backup_path, modified)
                        break

            return True

        except Exception as e:
            logger.error(f"Undo failed: {e}")
            return False

    def suggest_refactorings(
        self,
        file_path: Path,
    ) -> list[RefactorSuggestion]:
        """Suggest potential refactorings for a file.

        Args:
            file_path: Path to analyze

        Returns:
            List of suggestions
        """
        language = self._detect_language(file_path)
        analyzer = self.get_analyzer(language)

        if analyzer is None:
            return []

        try:
            source = file_path.read_text()
            return analyzer.suggest_refactorings(source, file_path)
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return []

    def suggest_refactorings_for_project(
        self,
        max_files: int = 100,
    ) -> dict[Path, list[RefactorSuggestion]]:
        """Suggest refactorings for all files in the project.

        Args:
            max_files: Maximum files to analyze

        Returns:
            Dict mapping files to suggestions
        """
        suggestions: dict[Path, list[RefactorSuggestion]] = {}
        files_processed = 0

        for file_path in self.project_root.rglob("*.py"):
            if files_processed >= max_files:
                break

            # Skip common non-source directories
            if any(
                part in file_path.parts
                for part in [".git", "__pycache__", "venv", "node_modules", ".tox"]
            ):
                continue

            file_suggestions = self.suggest_refactorings(file_path)
            if file_suggestions:
                suggestions[file_path] = file_suggestions

            files_processed += 1

        return suggestions

    def rename_symbol(
        self,
        file_path: Path,
        line: int,
        column: int,
        new_name: str,
        scope: RefactorScope = RefactorScope.PROJECT,
    ) -> RefactorResult:
        """Convenience method to rename a symbol.

        Args:
            file_path: File containing the symbol
            line: Line number
            column: Column number
            new_name: New name for the symbol
            scope: Refactoring scope

        Returns:
            RefactorResult
        """
        request = RefactorRequest(
            refactor_type=RefactorType.RENAME_SYMBOL,
            target=SourceLocation(
                file_path=file_path,
                start_line=line,
                start_column=column,
                end_line=line,
                end_column=column + 1,  # Will be adjusted by analyzer
            ),
            scope=scope,
            new_name=new_name,
        )
        return self.apply(request)

    def extract_function(
        self,
        file_path: Path,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        function_name: str,
    ) -> RefactorResult:
        """Convenience method to extract a function.

        Args:
            file_path: File containing the code
            start_line: Start line
            start_column: Start column
            end_line: End line
            end_column: End column
            function_name: Name for the new function

        Returns:
            RefactorResult
        """
        request = RefactorRequest(
            refactor_type=RefactorType.EXTRACT_FUNCTION,
            target=SourceLocation(
                file_path=file_path,
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
            ),
            extract_name=function_name,
        )
        return self.apply(request)

    def extract_variable(
        self,
        file_path: Path,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        variable_name: str,
    ) -> RefactorResult:
        """Convenience method to extract a variable.

        Args:
            file_path: File containing the expression
            start_line: Start line
            start_column: Start column
            end_line: End line
            end_column: End column
            variable_name: Name for the new variable

        Returns:
            RefactorResult
        """
        request = RefactorRequest(
            refactor_type=RefactorType.EXTRACT_VARIABLE,
            target=SourceLocation(
                file_path=file_path,
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
            ),
            extract_name=variable_name,
        )
        return self.apply(request)

    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file path."""
        ext = file_path.suffix.lower()
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
        }
        return ext_map.get(ext, "python")

    def _load_sources(
        self,
        target_file: Path,
        scope: RefactorScope,
    ) -> dict[Path, str]:
        """Load source files based on scope.

        Args:
            target_file: Primary target file
            scope: Scope determining which files to load

        Returns:
            Dict mapping paths to source content
        """
        sources: dict[Path, str] = {}

        if scope == RefactorScope.LOCAL:
            # Only target file
            try:
                sources[target_file] = target_file.read_text()
            except Exception as e:
                logger.error(f"Failed to read {target_file}: {e}")

        elif scope in (RefactorScope.PROJECT, RefactorScope.WORKSPACE):
            # All Python files in project
            language = self._detect_language(target_file)
            ext = f".{language}" if language != "python" else ".py"

            for file_path in self.project_root.rglob(f"*{ext}"):
                # Skip non-source directories
                if any(
                    part in file_path.parts
                    for part in [".git", "__pycache__", "venv", "node_modules"]
                ):
                    continue

                try:
                    sources[file_path] = file_path.read_text()
                except Exception as e:
                    logger.debug(f"Skipped {file_path}: {e}")

        return sources

    def _create_backups(self, files: list[Path]) -> list[Path]:
        """Create backups of files.

        Args:
            files: Files to back up

        Returns:
            List of backup paths
        """
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_paths = []

        for file_path in files:
            backup_name = f"{file_path.stem}.backup_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name

            try:
                shutil.copy(file_path, backup_path)
                backup_paths.append(backup_path)
            except Exception as e:
                logger.warning(f"Failed to backup {file_path}: {e}")

        return backup_paths


# Global manager singleton
_refactor_manager: Optional[RefactorManager] = None


def get_refactor_manager(
    project_root: Optional[Path] = None,
) -> RefactorManager:
    """Get the global refactor manager.

    Args:
        project_root: Project root directory

    Returns:
        RefactorManager instance
    """
    global _refactor_manager
    if _refactor_manager is None or (
        project_root and _refactor_manager.project_root != project_root
    ):
        _refactor_manager = RefactorManager(project_root=project_root)
    return _refactor_manager


def reset_refactor_manager() -> None:
    """Reset the global refactor manager."""
    global _refactor_manager
    _refactor_manager = None
