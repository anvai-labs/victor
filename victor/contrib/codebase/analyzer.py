# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Basic codebase-analysis contract projected from victor-codegraph v2."""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from victor.core.codegraph_adapter import parse_code, project_python_imports
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
    """Dependency-light analyzer whose semantic parsing is owned by CodeGraph."""

    LANGUAGE_MAP: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
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

    async def analyze_codebase(
        self,
        root_path: Path,
        include_patterns: List[str],
        exclude_patterns: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> CodebaseAnalysis:
        del kwargs
        included = {
            match
            for pattern in include_patterns
            for match in glob.glob(str(root_path / pattern), recursive=True)
            if Path(match).is_file()
        }
        excluded = {
            match
            for pattern in exclude_patterns or []
            for match in glob.glob(str(root_path / pattern), recursive=True)
        }
        files = sorted(included - excluded)
        languages: dict[str, int] = {}
        dependencies: dict[str, List[str]] = {}
        total_lines = 0

        for file_path_str in files:
            file_path = Path(file_path_str)
            try:
                parsed = await self.parse_file(file_path, repo_root=root_path)
            except Exception as exc:
                logger.warning("Error analyzing %s: %s", file_path, exc)
                continue
            total_lines += parsed.lines
            languages[parsed.language] = languages.get(parsed.language, 0) + 1
            file_dependencies = self._dependencies_for_parsed(parsed)
            if file_dependencies.external_packages:
                dependencies[file_path_str] = file_dependencies.external_packages

        return CodebaseAnalysis(
            root_path=root_path,
            files=files,
            total_files=len(files),
            total_lines=total_lines,
            languages=languages,
            dependencies=dependencies,
            structure={"root": str(root_path)},
            metadata={"parser": "victor_codegraph"},
        )

    async def parse_file(self, file_path: Path, **kwargs: Any) -> ParsedFile:
        repo_root = kwargs.get("repo_root")
        language = self._detect_language(file_path)
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return ParsedFile(
                file_path=file_path,
                language=language,
                lines=0,
                errors=[f"Cannot read file: {exc}"],
            )

        canonical = parse_code(
            content,
            file_path=str(file_path),
            language=language,
            repo_root=repo_root,
        )
        if canonical is None:
            return ParsedFile(
                file_path=file_path,
                language=language,
                lines=len(content.split("\n")),
                errors=["victor-codegraph parser unavailable"],
            )

        classes = [
            self._class_info(symbol)
            for symbol in canonical.symbols
            if symbol.symbol_type.name in {"CLASS", "INTERFACE", "TRAIT", "STRUCT", "ENUM"}
            and not symbol.scope_chain
        ]
        functions = [
            self._function_info(symbol)
            for symbol in canonical.symbols
            if symbol.symbol_type.name == "FUNCTION" and not symbol.scope_chain
        ]
        imports = [
            ImportInfo(
                module=item.module,
                names=list(item.names),
                alias=item.alias,
                is_from_import=item.is_from_import,
            )
            for item in project_python_imports(canonical.imports)
        ]
        errors = [
            diagnostic.message
            for diagnostic in canonical.diagnostics
            if diagnostic.severity == "error"
        ]
        return ParsedFile(
            file_path=file_path,
            language=language,
            lines=len(content.split("\n")),
            classes=classes,
            functions=functions,
            imports=imports,
            errors=errors,
        )

    async def get_dependencies(self, file_path: Path, **kwargs: Any) -> FileDependencies:
        del kwargs
        return self._dependencies_for_parsed(await self.parse_file(file_path))

    def get_analyzer_info(self) -> dict[str, Any]:
        return {
            "name": "BasicCodebaseAnalyzer",
            "version": "1.0.0",
            "capabilities": [
                "file_discovery",
                "language_detection",
                "victor_codegraph",
            ],
        }

    def _detect_language(self, file_path: Path) -> str:
        return self.LANGUAGE_MAP.get(file_path.suffix.lower(), "unknown")

    @staticmethod
    def _class_info(symbol: Any) -> ClassInfo:
        modifiers = list(symbol.modifiers)
        bases = next(
            (
                [base.strip() for base in item[8:-1].split(",") if base.strip()]
                for item in modifiers
                if item.startswith("extends(") and item.endswith(")")
            ),
            [],
        )
        return ClassInfo(
            name=symbol.simple_name,
            line_number=symbol.location.start_line,
            end_line_number=symbol.location.end_line,
            bases=bases,
            decorators=[item[1:] for item in modifiers if item.startswith("@")],
            docstring=symbol.documentation,
        )

    @staticmethod
    def _function_info(symbol: Any) -> FunctionInfo:
        modifiers = list(symbol.modifiers)
        return FunctionInfo(
            name=symbol.simple_name,
            line_number=symbol.location.start_line,
            end_line_number=symbol.location.end_line,
            parameters=[str(parameter.get("name", "")) for parameter in symbol.parameters],
            return_type=symbol.return_type,
            decorators=[item[1:] for item in modifiers if item.startswith("@")],
            docstring=symbol.documentation,
            is_async="async" in modifiers,
        )

    @staticmethod
    def _dependencies_for_parsed(parsed: ParsedFile) -> FileDependencies:
        external: list[str] = []
        internal: list[str] = []
        for imported in parsed.imports:
            if imported.module.startswith("."):
                internal.append(imported.module)
            elif imported.module.split(".", 1)[0] not in sys.stdlib_module_names:
                external.append(imported.module)
        return FileDependencies(
            file_path=parsed.file_path,
            imports=parsed.imports,
            external_packages=external,
            internal_modules=internal,
        )


__all__ = ["BasicCodebaseAnalyzer"]
