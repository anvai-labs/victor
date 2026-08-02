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

"""
Basic codebase analyzer implementation.

This module provides a simple file-based codebase analyzer that can be used
as a default implementation when more advanced parsers (like tree-sitter)
are not available.

SOLID Principles:
- SRP: BasicCodebaseAnalyzer only handles basic analysis
- OCP: Extensible through protocol implementation
- LSP: Implements CodebaseAnalyzerProtocol completely
- ISP: Focused on basic file operations
- DIP: No dependencies on concrete implementations

Usage:
    from victor.contrib.codebase import BasicCodebaseAnalyzer

    analyzer = BasicCodebaseAnalyzer()
    analysis = await analyzer.analyze_codebase(
        root_path=Path("/path/to/code"),
        include_patterns=["**/*.py"],
    )
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from victor.framework.vertical_protocols import (
    ClassInfo,
    CodebaseAnalysis,
    CodebaseAnalyzerProtocol,
    FileDependencies,
    FunctionInfo,
    ImportInfo,
    ParsedFile,
)

logger = logging.getLogger(__name__)


class BasicCodebaseAnalyzer(CodebaseAnalyzerProtocol):
    """
    Basic file-based codebase analyzer.

    This analyzer provides simple file discovery and pattern-based
    parsing without requiring heavy dependencies like tree-sitter.

    Features:
    - File discovery via glob patterns
    - Basic language detection via file extension
    - Pattern-based extraction (classes, functions, imports)
    - Line counting
    - Dependency extraction

    Example:
        analyzer = BasicCodebaseAnalyzer()
        analysis = await analyzer.analyze_codebase(
            root_path=Path("/path/to/code"),
            include_patterns=["**/*.py", "**/*.js"],
            exclude_patterns=["**/test_*.py"],
        )
    """

    # Language mappings
    LANGUAGE_MAP: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".sh": "shell",
        ".bash": "shell",
        ".zsh": "shell",
        ".fish": "shell",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".md": "markdown",
        ".txt": "text",
    }

    # Patterns for Python extraction
    CLASS_PATTERN = re.compile(r"^class\s+(\w+)(?:\(([^)]+)\))?:")
    FUNCTION_PATTERN = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)")
    IMPORT_PATTERN = re.compile(r"^(?:from\s+([^\s]+)\s+)?import\s+(\S+)")
    DECORATOR_PATTERN = re.compile(r"^@\s*(\w+)")

    def __init__(self) -> None:
        """Initialize the analyzer."""

    async def analyze_codebase(
        self,
        root_path: Path,
        include_patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CodebaseAnalysis:
        """Analyze a codebase.

        Args:
            root_path: Root directory of the codebase
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            **kwargs: Additional implementation-specific options

        Returns:
            CodebaseAnalysis with files, structure, dependencies, etc.
        """
        import glob

        files: List[str] = []
        languages: dict[str, int] = {}
        total_lines = 0
        all_dependencies: dict[str, List[str]] = {}

        # Find all matching files
        included_files: set[str] = set()
        for pattern in include_patterns:
            matches = glob.glob(str(root_path / pattern), recursive=True)
            included_files.update(matches)

        # Remove excluded files
        if exclude_patterns:
            excluded_files: set[str] = set()
            for pattern in exclude_patterns:
                matches = glob.glob(str(root_path / pattern), recursive=True)
                excluded_files.update(matches)
            included_files -= excluded_files

        files = sorted(list(included_files))

        # Analyze each file
        for file_path_str in files:
            file_path = Path(file_path_str)
            try:
                parsed = await self.parse_file(file_path)
                total_lines += parsed.lines

                # Count by language
                if parsed.language:
                    languages[parsed.language] = languages.get(parsed.language, 0) + 1

                # Collect dependencies
                deps = await self.get_dependencies(file_path)
                if deps.external_packages:
                    all_dependencies[file_path_str] = deps.external_packages

            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

        return CodebaseAnalysis(
            root_path=root_path,
            files=files,
            total_files=len(files),
            total_lines=total_lines,
            languages=languages,
            dependencies=all_dependencies,
            structure={"root": str(root_path)},
        )

    async def parse_file(self, file_path: Path, **kwargs: Any) -> ParsedFile:
        """Parse a single source file.

        Args:
            file_path: Path to the file to parse
            **kwargs: Additional implementation-specific options

        Returns:
            ParsedFile with syntax tree and extracted information
        """
        # Detect language from extension
        language = self._detect_language(file_path)

        # Read file
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
        except Exception as e:
            return ParsedFile(
                file_path=file_path,
                language=language,
                lines=0,
                errors=[f"Cannot read file: {e}"],
            )

        # Extract information based on language
        classes: List[ClassInfo] = []
        functions: List[FunctionInfo] = []
        imports: List[ImportInfo] = []

        if language == "python":
            classes, functions, imports = self._parse_python(content, lines)

        return ParsedFile(
            file_path=file_path,
            language=language,
            lines=len(lines),
            classes=classes,
            functions=functions,
            imports=imports,
        )

    async def get_dependencies(self, file_path: Path, **kwargs: Any) -> FileDependencies:
        """Get dependencies for a file.

        Args:
            file_path: Path to the file
            **kwargs: Additional implementation-specific options

        Returns:
            FileDependencies with imports and requirements
        """
        parsed = await self.parse_file(file_path)

        external_packages: List[str] = []
        internal_modules: List[str] = []

        for imp in parsed.imports:
            # Classify as external or internal
            if imp.module.startswith("."):
                # Relative import - internal
                internal_modules.append(imp.module)
            elif imp.module.split(".")[0] in ["os", "sys", "json", "re", "pathlib"]:
                # Standard library - external
                pass
            else:
                # Could be external or internal - conservative classification
                external_packages.append(imp.module)

        return FileDependencies(
            file_path=file_path,
            imports=parsed.imports,
            external_packages=external_packages,
            internal_modules=internal_modules,
        )

    def get_analyzer_info(self) -> dict[str, Any]:
        """Get analyzer metadata."""
        return {
            "name": "BasicCodebaseAnalyzer",
            "version": "1.0.0",
            "capabilities": [
                "file_discovery",
                "language_detection",
                "pattern_based_extraction",
            ],
        }

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(suffix, "unknown")

    def _parse_python(
        self, content: str, lines: List[str]
    ) -> tuple[List[ClassInfo], List[FunctionInfo], List[ImportInfo]]:
        """Parse Python source code.

        Args:
            content: File content
            lines: Lines of the file

        Returns:
            Tuple of (classes, functions, imports)
        """
        classes: List[ClassInfo] = []
        functions: List[FunctionInfo] = []
        imports: List[ImportInfo] = []
        decorators: List[str] = []

        for line_num, line in enumerate(lines):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith("#"):
                continue

            # Check for decorator
            dec_match = self.DECORATOR_PATTERN.match(line)
            if dec_match:
                decorators.append(dec_match.group(1))
                continue

            # Check for class definition
            class_match = self.CLASS_PATTERN.match(line)
            if class_match:
                name = class_match.group(1)
                bases = class_match.group(2).split(",") if class_match.group(2) else []
                classes.append(
                    ClassInfo(
                        name=name,
                        line_number=line_num,
                        bases=[b.strip() for b in bases],
                        decorators=decorators.copy(),
                    )
                )
                decorators.clear()
                continue

            # Check for function definition
            func_match = self.FUNCTION_PATTERN.match(line)
            if func_match:
                name = func_match.group(1)
                params = func_match.group(2).split(",") if func_match.group(2) else []
                functions.append(
                    FunctionInfo(
                        name=name,
                        line_number=line_num,
                        parameters=[p.strip() for p in params],
                        is_async=line.strip().startswith("async"),
                        decorators=decorators.copy(),
                    )
                )
                decorators.clear()
                continue

            # Check for import
            import_match = self.IMPORT_PATTERN.match(line)
            if import_match:
                from_module = import_match.group(
                    1
                )  # None for "import os", "pathlib" for "from pathlib import..."
                import_names = import_match.group(2).split(",")

                # For "import os": module="os", names=[]
                # For "from pathlib import Path": module="pathlib", names=["Path"]
                if from_module:
                    # "from X import Y" style
                    module = from_module
                    names = [n.strip() for n in import_names]
                    is_from = True
                else:
                    # "import X" style - the first name is the module
                    first_name = import_names[0].strip() if import_names else ""
                    module = first_name
                    names = [n.strip() for n in import_names[1:]] if len(import_names) > 1 else []
                    is_from = False

                imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        line_number=line_num,
                        is_from_import=is_from,
                    )
                )
                continue

        return classes, functions, imports


__all__ = ["BasicCodebaseAnalyzer"]
