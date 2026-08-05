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

"""File type detection protocols and implementations.

This module provides generic file type detection based on:
- File extensions (.py, .js, .yaml, etc.)
- File names (Dockerfile, Makefile, etc.)
- Shebangs (#!/usr/bin/python, etc.)
- Content analysis

Design Principles:
- SRP: Only handles file type detection, not language tooling
- OCP: FileTypeRegistry is open for extension via register()
- DIP: FileTypeDetector protocol allows custom implementations

Example usage:
    from victor.processing.file_types import FileTypeRegistry, detect_file_type

    # Detect file type
    file_type = detect_file_type(Path("script.py"))
    print(f"Type: {file_type.name}, Category: {file_type.category}")

    # Register custom type
    registry = FileTypeRegistry.get_instance()
    registry.register(FileType(
        name="custom",
        extensions=[".custom"],
        category=FileCategory.DATA,
    ))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable


class FileCategory(Enum):
    """High-level file categories for cross-vertical use."""

    # Programming languages
    CODE = "code"

    # Configuration files
    CONFIG = "config"

    # Data files
    DATA = "data"

    # Documentation
    DOCS = "docs"

    # Build/Infrastructure
    BUILD = "build"

    # Media
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

    # Archives
    ARCHIVE = "archive"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class FileType:
    """Definition of a file type.

    Attributes:
        name: Canonical name (e.g., "python", "yaml", "dockerfile")
        display_name: Human-readable name (e.g., "Python", "YAML")
        extensions: File extensions including dot (e.g., [".py", ".pyw"])
        filenames: Exact filenames (e.g., ["Dockerfile", "Makefile"])
        shebangs: Shebang patterns (e.g., ["python", "python3"])
        mime_types: MIME types (e.g., ["text/x-python"])
        category: High-level category
        aliases: Alternative names
    """

    name: str
    display_name: str = ""
    extensions: List[str] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)
    shebangs: List[str] = field(default_factory=list)
    mime_types: List[str] = field(default_factory=list)
    category: FileCategory = FileCategory.UNKNOWN
    aliases: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.title()

    def matches_extension(self, path: Path) -> bool:
        """Check if path matches any extension."""
        return path.suffix.lower() in [e.lower() for e in self.extensions]

    def matches_filename(self, path: Path) -> bool:
        """Check if path matches any filename."""
        return path.name in self.filenames

    def matches_shebang(self, first_line: str) -> bool:
        """Check if first line matches any shebang."""
        if not first_line.startswith("#!"):
            return False
        for shebang in self.shebangs:
            if shebang in first_line:
                return True
        return False


@runtime_checkable
class FileTypeDetector(Protocol):
    """Protocol for file type detection.

    Implementations can provide custom detection logic beyond
    simple extension/filename matching.
    """

    def detect(self, path: Path, content: Optional[str] = None) -> Optional[FileType]:
        """Detect file type from path and optional content.

        Args:
            path: File path
            content: Optional file content for content-based detection

        Returns:
            FileType if detected, None otherwise
        """
        ...

    def confidence(self, path: Path, content: Optional[str] = None) -> float:
        """Get confidence score for detection.

        Args:
            path: File path
            content: Optional file content

        Returns:
            Confidence score 0.0 to 1.0
        """
        ...


class FileTypeRegistry:
    """Registry for file types with detection capabilities.

    Implements the Registry pattern for extensible file type detection.
    Uses singleton pattern for global access.

    Example:
        registry = FileTypeRegistry.get_instance()
        file_type = registry.detect(Path("app.py"))
    """

    _instance: Optional["FileTypeRegistry"] = None

    def __init__(self):
        """Initialize the registry with built-in types."""
        self._types: Dict[str, FileType] = {}
        self._extension_map: Dict[str, str] = {}  # ext -> type name
        self._filename_map: Dict[str, str] = {}  # filename -> type name
        self._register_builtin_types()

    @classmethod
    def get_instance(cls) -> "FileTypeRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def register(self, file_type: FileType) -> None:
        """Register a file type.

        Args:
            file_type: FileType to register
        """
        self._types[file_type.name] = file_type

        for ext in file_type.extensions:
            self._extension_map[ext.lower()] = file_type.name

        for filename in file_type.filenames:
            self._filename_map[filename] = file_type.name

        for alias in file_type.aliases:
            self._types[alias] = file_type

    def get(self, name: str) -> Optional[FileType]:
        """Get a file type by name.

        Args:
            name: File type name or alias

        Returns:
            FileType or None
        """
        return self._types.get(name.lower())

    def detect(self, path: Path, content: Optional[str] = None) -> Optional[FileType]:
        """Detect file type from path and optional content.

        Detection order:
        1. Exact filename match (Dockerfile, Makefile)
        2. Extension match (.py, .js)
        3. Shebang match (if content provided)

        Args:
            path: File path
            content: Optional file content

        Returns:
            FileType if detected, None otherwise
        """
        # Check filename first (highest priority)
        if path.name in self._filename_map:
            return self._types[self._filename_map[path.name]]

        # Check extension
        ext = path.suffix.lower()
        if ext in self._extension_map:
            return self._types[self._extension_map[ext]]

        # Check shebang if content available
        if content:
            first_line = content.split("\n")[0] if content else ""
            if first_line.startswith("#!"):
                for file_type in self._types.values():
                    if file_type.matches_shebang(first_line):
                        return file_type

        return None

    def detect_from_content(
        self, content: str, filename: Optional[str] = None
    ) -> Optional[FileType]:
        """Detect file type from content.

        Args:
            content: File content
            filename: Optional filename hint

        Returns:
            FileType if detected, None otherwise
        """
        if filename:
            return self.detect(Path(filename), content)

        # Try shebang detection
        first_line = content.split("\n")[0] if content else ""
        if first_line.startswith("#!"):
            for file_type in self._types.values():
                if file_type.matches_shebang(first_line):
                    return file_type

        return None

    def get_by_category(self, category: FileCategory) -> List[FileType]:
        """Get all file types in a category.

        Args:
            category: FileCategory to filter by

        Returns:
            List of matching FileTypes
        """
        return [ft for ft in self._types.values() if ft.category == category]

    def get_by_extension(self, extension: str) -> Optional[FileType]:
        """Get file type by extension.

        Args:
            extension: File extension (with or without dot)

        Returns:
            FileType or None
        """
        ext = extension if extension.startswith(".") else f".{extension}"
        name = self._extension_map.get(ext.lower())
        return self._types.get(name) if name else None

    def list_all(self) -> List[FileType]:
        """List all registered file types.

        Returns:
            List of all FileTypes (excluding aliases)
        """
        seen = set()
        result = []
        for ft in self._types.values():
            if ft.name not in seen:
                seen.add(ft.name)
                result.append(ft)
        return result

    def _register_builtin_types(self) -> None:
        """Register built-in file types."""
        # Programming Languages
        self.register(
            FileType(
                name="python",
                display_name="Python",
                extensions=[".py", ".pyw", ".pyi", ".pyx"],
                filenames=["SConstruct", "SConscript"],
                shebangs=["python", "python3"],
                mime_types=["text/x-python"],
                category=FileCategory.CODE,
                aliases=["py"],
            )
        )

        self.register(
            FileType(
                name="javascript",
                display_name="JavaScript",
                extensions=[".js", ".mjs", ".cjs"],
                shebangs=["node", "nodejs"],
                mime_types=["text/javascript", "application/javascript"],
                category=FileCategory.CODE,
                aliases=["js"],
            )
        )

        self.register(
            FileType(
                name="typescript",
                display_name="TypeScript",
                extensions=[".ts", ".tsx", ".mts", ".cts"],
                mime_types=["text/typescript"],
                category=FileCategory.CODE,
                aliases=["ts"],
            )
        )

        self.register(
            FileType(
                name="rust",
                display_name="Rust",
                extensions=[".rs"],
                mime_types=["text/x-rust"],
                category=FileCategory.CODE,
                aliases=["rs"],
            )
        )

        self.register(
            FileType(
                name="go",
                display_name="Go",
                extensions=[".go"],
                mime_types=["text/x-go"],
                category=FileCategory.CODE,
                aliases=["golang"],
            )
        )

        self.register(
            FileType(
                name="java",
                display_name="Java",
                extensions=[".java"],
                mime_types=["text/x-java"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="kotlin",
                display_name="Kotlin",
                extensions=[".kt", ".kts"],
                mime_types=["text/x-kotlin"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="swift",
                display_name="Swift",
                extensions=[".swift"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="c",
                display_name="C",
                extensions=[".c", ".h"],
                mime_types=["text/x-c"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="cpp",
                display_name="C++",
                extensions=[".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
                mime_types=["text/x-c++"],
                category=FileCategory.CODE,
                aliases=["c++", "cxx"],
            )
        )

        self.register(
            FileType(
                name="csharp",
                display_name="C#",
                extensions=[".cs"],
                mime_types=["text/x-csharp"],
                category=FileCategory.CODE,
                aliases=["cs", "c#"],
            )
        )

        self.register(
            FileType(
                name="ruby",
                display_name="Ruby",
                extensions=[".rb", ".rake", ".gemspec"],
                filenames=["Rakefile", "Gemfile"],
                shebangs=["ruby"],
                mime_types=["text/x-ruby"],
                category=FileCategory.CODE,
                aliases=["rb"],
            )
        )

        self.register(
            FileType(
                name="php",
                display_name="PHP",
                extensions=[".php", ".phtml", ".php3", ".php4", ".php5"],
                shebangs=["php"],
                mime_types=["text/x-php"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="shell",
                display_name="Shell",
                extensions=[".sh", ".bash", ".zsh"],
                shebangs=["sh", "bash", "zsh", "/bin/sh", "/bin/bash"],
                mime_types=["text/x-shellscript"],
                category=FileCategory.CODE,
                aliases=["bash", "sh", "zsh"],
            )
        )

        self.register(
            FileType(
                name="powershell",
                display_name="PowerShell",
                extensions=[".ps1", ".psm1", ".psd1"],
                mime_types=["text/x-powershell"],
                category=FileCategory.CODE,
                aliases=["ps1"],
            )
        )

        # Configuration Files
        self.register(
            FileType(
                name="yaml",
                display_name="YAML",
                extensions=[".yaml", ".yml"],
                mime_types=["text/yaml", "application/x-yaml"],
                category=FileCategory.CONFIG,
                aliases=["yml"],
            )
        )

        self.register(
            FileType(
                name="json",
                display_name="JSON",
                extensions=[".json", ".jsonc"],
                mime_types=["application/json"],
                category=FileCategory.CONFIG,
            )
        )

        self.register(
            FileType(
                name="toml",
                display_name="TOML",
                extensions=[".toml"],
                mime_types=["application/toml"],
                category=FileCategory.CONFIG,
            )
        )

        self.register(
            FileType(
                name="ini",
                display_name="INI",
                extensions=[".ini", ".cfg", ".conf"],
                category=FileCategory.CONFIG,
                aliases=["cfg", "conf"],
            )
        )

        self.register(
            FileType(
                name="xml",
                display_name="XML",
                extensions=[".xml", ".xsd", ".xsl", ".xslt"],
                mime_types=["text/xml", "application/xml"],
                category=FileCategory.CONFIG,
            )
        )

        self.register(
            FileType(
                name="properties",
                display_name="Properties",
                extensions=[".properties"],
                category=FileCategory.CONFIG,
            )
        )

        self.register(
            FileType(
                name="env",
                display_name="Environment",
                extensions=[".env"],
                filenames=[".env", ".env.local", ".env.development", ".env.production"],
                category=FileCategory.CONFIG,
                aliases=["dotenv"],
            )
        )

        # Data Files
        self.register(
            FileType(
                name="csv",
                display_name="CSV",
                extensions=[".csv"],
                mime_types=["text/csv"],
                category=FileCategory.DATA,
            )
        )

        self.register(
            FileType(
                name="tsv",
                display_name="TSV",
                extensions=[".tsv"],
                mime_types=["text/tab-separated-values"],
                category=FileCategory.DATA,
            )
        )

        self.register(
            FileType(
                name="parquet",
                display_name="Parquet",
                extensions=[".parquet"],
                mime_types=["application/vnd.apache.parquet"],
                category=FileCategory.DATA,
            )
        )

        self.register(
            FileType(
                name="sql",
                display_name="SQL",
                extensions=[".sql"],
                mime_types=["text/x-sql"],
                category=FileCategory.DATA,
            )
        )

        # Documentation
        self.register(
            FileType(
                name="markdown",
                display_name="Markdown",
                extensions=[".md", ".markdown", ".mdown"],
                mime_types=["text/markdown"],
                category=FileCategory.DOCS,
                aliases=["md"],
            )
        )

        self.register(
            FileType(
                name="rst",
                display_name="reStructuredText",
                extensions=[".rst"],
                mime_types=["text/x-rst"],
                category=FileCategory.DOCS,
                aliases=["restructuredtext"],
            )
        )

        self.register(
            FileType(
                name="text",
                display_name="Plain Text",
                extensions=[".txt", ".text"],
                mime_types=["text/plain"],
                category=FileCategory.DOCS,
                aliases=["txt", "plaintext"],
            )
        )

        # Build Files
        self.register(
            FileType(
                name="dockerfile",
                display_name="Dockerfile",
                filenames=["Dockerfile", "Dockerfile.dev", "Dockerfile.prod"],
                category=FileCategory.BUILD,
                aliases=["docker"],
            )
        )

        self.register(
            FileType(
                name="makefile",
                display_name="Makefile",
                filenames=["Makefile", "makefile", "GNUmakefile"],
                category=FileCategory.BUILD,
                aliases=["make"],
            )
        )

        self.register(
            FileType(
                name="cmake",
                display_name="CMake",
                extensions=[".cmake"],
                filenames=["CMakeLists.txt"],
                category=FileCategory.BUILD,
            )
        )

        # Web
        self.register(
            FileType(
                name="html",
                display_name="HTML",
                extensions=[".html", ".htm", ".xhtml"],
                mime_types=["text/html"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="css",
                display_name="CSS",
                extensions=[".css"],
                mime_types=["text/css"],
                category=FileCategory.CODE,
            )
        )

        self.register(
            FileType(
                name="scss",
                display_name="SCSS",
                extensions=[".scss"],
                category=FileCategory.CODE,
                aliases=["sass"],
            )
        )

        # Images
        self.register(
            FileType(
                name="png",
                display_name="PNG Image",
                extensions=[".png"],
                mime_types=["image/png"],
                category=FileCategory.IMAGE,
            )
        )

        self.register(
            FileType(
                name="jpeg",
                display_name="JPEG Image",
                extensions=[".jpg", ".jpeg"],
                mime_types=["image/jpeg"],
                category=FileCategory.IMAGE,
                aliases=["jpg"],
            )
        )

        self.register(
            FileType(
                name="svg",
                display_name="SVG",
                extensions=[".svg"],
                mime_types=["image/svg+xml"],
                category=FileCategory.IMAGE,
            )
        )

        self.register(
            FileType(
                name="gif",
                display_name="GIF",
                extensions=[".gif"],
                mime_types=["image/gif"],
                category=FileCategory.IMAGE,
            )
        )

        # Archives
        self.register(
            FileType(
                name="zip",
                display_name="ZIP Archive",
                extensions=[".zip"],
                mime_types=["application/zip"],
                category=FileCategory.ARCHIVE,
            )
        )

        self.register(
            FileType(
                name="tar",
                display_name="TAR Archive",
                extensions=[".tar", ".tar.gz", ".tgz", ".tar.bz2"],
                mime_types=["application/x-tar"],
                category=FileCategory.ARCHIVE,
            )
        )


# Convenience functions


def detect_file_type(path: Path, content: Optional[str] = None) -> Optional[FileType]:
    """Detect file type from path.

    Args:
        path: File path
        content: Optional file content for shebang detection

    Returns:
        FileType if detected, None otherwise
    """
    return FileTypeRegistry.get_instance().detect(path, content)


def get_file_category(path: Path) -> FileCategory:
    """Get file category for a path.

    Args:
        path: File path

    Returns:
        FileCategory (UNKNOWN if not detected)
    """
    file_type = detect_file_type(path)
    return file_type.category if file_type else FileCategory.UNKNOWN


def is_code_file(path: Path) -> bool:
    """Check if path is a code file.

    Args:
        path: File path

    Returns:
        True if file is in CODE category
    """
    return get_file_category(path) == FileCategory.CODE


def is_config_file(path: Path) -> bool:
    """Check if path is a config file.

    Args:
        path: File path

    Returns:
        True if file is in CONFIG category
    """
    return get_file_category(path) == FileCategory.CONFIG


def is_data_file(path: Path) -> bool:
    """Check if path is a data file.

    Args:
        path: File path

    Returns:
        True if file is in DATA category
    """
    return get_file_category(path) == FileCategory.DATA


__all__ = [
    # Types
    "FileCategory",
    "FileType",
    "FileTypeDetector",
    # Registry
    "FileTypeRegistry",
    # Functions
    "detect_file_type",
    "get_file_category",
    "is_code_file",
    "is_config_file",
    "is_data_file",
]
