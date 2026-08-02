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

"""Documentation generation manager.

Provides high-level API for automated documentation generation.
"""

import logging
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

from victor_coding.docgen.formatter import BaseFormatter, get_formatter
from victor_coding.docgen.parser import CodeAnalyzer
from victor_coding.docgen.protocol import (
    DocConfig,
    DocFormat,
    DocGenResult,
    GeneratedDoc,
)

logger = logging.getLogger(__name__)


class DocGenManager:
    """High-level manager for documentation generation.

    Orchestrates code analysis and documentation formatting.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config: Optional[DocConfig] = None,
    ):
        """Initialize the manager.

        Args:
            project_root: Root directory of the project
            config: Default generation configuration
        """
        self.project_root = project_root or Path.cwd()
        self.config = config or DocConfig()

        # Initialize components
        self.analyzer = CodeAnalyzer(self.config.input_style)
        self._formatters: dict[DocFormat, BaseFormatter] = {}

    def get_formatter(self, format: DocFormat) -> BaseFormatter:
        """Get or create a formatter.

        Args:
            format: Output format

        Returns:
            Formatter instance
        """
        if format not in self._formatters:
            self._formatters[format] = get_formatter(format)
        return self._formatters[format]

    def generate_for_file(
        self,
        file_path: Path,
        config: Optional[DocConfig] = None,
        write_file: bool = True,
    ) -> DocGenResult:
        """Generate documentation for a single file.

        Args:
            file_path: Path to the Python file
            config: Generation configuration
            write_file: Whether to write the output file

        Returns:
            DocGenResult
        """
        start_time = time.time()
        config = config or self.config
        result = DocGenResult(success=False)

        # Analyze file
        module_doc = self.analyzer.analyze_file(file_path)
        if not module_doc:
            result.errors.append(f"Failed to analyze {file_path}")
            return result

        # Generate documentation
        formatter = self.get_formatter(config.output_format)
        content = formatter.format_module(module_doc, config)

        # Determine output path
        output_path = self._get_output_path(file_path, config)

        # Create generated doc
        generated = GeneratedDoc(
            path=output_path,
            content=content,
            format=config.output_format,
        )

        # Write if requested
        if write_file:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content)
                logger.info(f"Wrote documentation: {output_path}")
            except Exception as e:
                result.errors.append(f"Failed to write {output_path}: {e}")

        result.generated_files.append(generated)
        result.modules_documented = 1
        result.classes_documented = len(module_doc.classes)
        result.functions_documented = len(module_doc.functions)
        result.success = True
        result.duration_ms = (time.time() - start_time) * 1000

        return result

    def generate_for_directory(
        self,
        directory: Path,
        config: Optional[DocConfig] = None,
        write_files: bool = True,
        recursive: bool = True,
        exclude_patterns: Optional[list[str]] = None,
    ) -> DocGenResult:
        """Generate documentation for all Python files in a directory.

        Args:
            directory: Directory to scan
            config: Generation configuration
            write_files: Whether to write output files
            recursive: Whether to recurse into subdirectories
            exclude_patterns: Glob patterns to exclude

        Returns:
            DocGenResult
        """
        start_time = time.time()
        config = config or self.config
        result = DocGenResult(success=True)
        exclude_patterns = exclude_patterns or [
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/node_modules/**",
            "**/test_*.py",
            "**/*_test.py",
        ]

        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        files = list(directory.glob(pattern))

        for file_path in files:
            # Check exclusions
            if self._should_exclude(file_path, exclude_patterns):
                continue

            file_result = self.generate_for_file(
                file_path,
                config=config,
                write_file=write_files,
            )

            # Merge results
            result.generated_files.extend(file_result.generated_files)
            result.modules_documented += file_result.modules_documented
            result.classes_documented += file_result.classes_documented
            result.functions_documented += file_result.functions_documented
            result.errors.extend(file_result.errors)
            result.warnings.extend(file_result.warnings)

            if file_result.errors:
                result.success = False

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def generate_for_package(
        self,
        package_path: Path,
        config: Optional[DocConfig] = None,
        write_files: bool = True,
    ) -> DocGenResult:
        """Generate documentation for a Python package.

        Args:
            package_path: Path to the package directory
            config: Generation configuration
            write_files: Whether to write output files

        Returns:
            DocGenResult with package documentation
        """
        config = config or self.config

        # Generate for all modules in package
        result = self.generate_for_directory(
            package_path,
            config=config,
            write_files=write_files,
            recursive=True,
        )

        # Generate index if configured
        if config.include_index and write_files:
            index_doc = self._generate_index(package_path, result, config)
            if index_doc:
                result.generated_files.append(index_doc)

        return result

    def generate_index(
        self,
        directory: Path,
        config: Optional[DocConfig] = None,
    ) -> Optional[str]:
        """Generate an index document for a directory.

        Args:
            directory: Directory containing documentation
            config: Generation configuration

        Returns:
            Index document content
        """
        config = config or self.config
        self.get_formatter(config.output_format)

        lines = []

        if config.output_format == DocFormat.MARKDOWN:
            lines.append(f"# {directory.name} Documentation")
            lines.append("")
            lines.append("## Modules")
            lines.append("")

            # Find all documented modules
            for py_file in sorted(directory.glob("**/*.py")):
                if py_file.name.startswith("_"):
                    continue

                module_name = py_file.stem
                rel_path = py_file.relative_to(directory)
                doc_path = rel_path.with_suffix(".md")

                lines.append(f"- [{module_name}]({doc_path})")

            lines.append("")

        elif config.output_format == DocFormat.HTML:
            lines.append("<!DOCTYPE html>")
            lines.append("<html><head>")
            lines.append(f"<title>{directory.name} Documentation</title>")
            lines.append("</head><body>")
            lines.append(f"<h1>{directory.name} Documentation</h1>")
            lines.append("<h2>Modules</h2><ul>")

            for py_file in sorted(directory.glob("**/*.py")):
                if py_file.name.startswith("_"):
                    continue

                module_name = py_file.stem
                rel_path = py_file.relative_to(directory)
                doc_path = rel_path.with_suffix(".html")

                lines.append(f'<li><a href="{doc_path}">{module_name}</a></li>')

            lines.append("</ul></body></html>")

        return "\n".join(lines) if lines else None

    def preview_documentation(
        self,
        file_path: Path,
        config: Optional[DocConfig] = None,
    ) -> Optional[str]:
        """Preview documentation without writing.

        Args:
            file_path: Source file
            config: Generation configuration

        Returns:
            Generated documentation content
        """
        config = config or self.config

        module_doc = self.analyzer.analyze_file(file_path)
        if not module_doc:
            return None

        formatter = self.get_formatter(config.output_format)
        return formatter.format_module(module_doc, config)

    def analyze_documentation_coverage(
        self,
        directory: Path,
    ) -> dict[str, dict[str, int]]:
        """Analyze documentation coverage for a directory.

        Args:
            directory: Directory to analyze

        Returns:
            Coverage statistics by file
        """
        coverage = {}

        for py_file in directory.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            module_doc = self.analyzer.analyze_file(py_file)
            if not module_doc:
                continue

            stats = {
                "functions_total": len(module_doc.functions),
                "functions_documented": sum(1 for f in module_doc.functions if f.description),
                "classes_total": len(module_doc.classes),
                "classes_documented": sum(1 for c in module_doc.classes if c.description),
                "methods_total": sum(len(c.methods) for c in module_doc.classes),
                "methods_documented": sum(
                    sum(1 for m in c.methods if m.description) for c in module_doc.classes
                ),
            }

            total = stats["functions_total"] + stats["classes_total"] + stats["methods_total"]
            documented = (
                stats["functions_documented"]
                + stats["classes_documented"]
                + stats["methods_documented"]
            )

            stats["coverage_percent"] = (documented / total * 100) if total > 0 else 100

            coverage[str(py_file)] = stats

        return coverage

    def _get_output_path(self, source_path: Path, config: DocConfig) -> Path:
        """Determine output path for documentation."""
        ext_map = {
            DocFormat.MARKDOWN: ".md",
            DocFormat.HTML: ".html",
            DocFormat.RST: ".rst",
            DocFormat.PLAINTEXT: ".txt",
        }

        ext = ext_map.get(config.output_format, ".md")

        if config.output_dir:
            output_dir = config.output_dir
        else:
            output_dir = source_path.parent / "docs"

        return output_dir / f"{source_path.stem}{ext}"

    def _should_exclude(self, file_path: Path, patterns: list[str]) -> bool:
        """Check if a file should be excluded."""
        str_path = str(file_path)
        for pattern in patterns:
            if fnmatch(str_path, pattern):
                return True
        return False

    def _generate_index(
        self,
        package_path: Path,
        result: DocGenResult,
        config: DocConfig,
    ) -> Optional[GeneratedDoc]:
        """Generate index document."""
        content = self.generate_index(package_path, config)
        if not content:
            return None

        ext_map = {
            DocFormat.MARKDOWN: ".md",
            DocFormat.HTML: ".html",
            DocFormat.RST: ".rst",
        }

        ext = ext_map.get(config.output_format, ".md")
        output_dir = config.output_dir or (package_path / "docs")
        index_path = output_dir / f"index{ext}"

        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_text(content)
            logger.info(f"Wrote index: {index_path}")
        except Exception as e:
            logger.error(f"Failed to write index: {e}")
            return None

        return GeneratedDoc(
            path=index_path,
            content=content,
            format=config.output_format,
        )


# Global manager singleton
_docgen_manager: Optional[DocGenManager] = None


def get_docgen_manager(
    project_root: Optional[Path] = None,
    config: Optional[DocConfig] = None,
) -> DocGenManager:
    """Get the global documentation generation manager.

    Args:
        project_root: Project root directory
        config: Default configuration

    Returns:
        DocGenManager instance
    """
    global _docgen_manager
    if _docgen_manager is None or (project_root and _docgen_manager.project_root != project_root):
        _docgen_manager = DocGenManager(project_root=project_root, config=config)
    return _docgen_manager


def reset_docgen_manager() -> None:
    """Reset the global manager."""
    global _docgen_manager
    _docgen_manager = None
