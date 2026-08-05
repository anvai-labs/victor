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

"""Project manifest for structured codebase context.

This module provides a structured representation of a project's key components,
enabling better context for LLMs when analyzing or modifying code.

Design Pattern: Builder + Repository
===================================
ProjectManifest acts as a repository of project knowledge, built incrementally
from various sources (file system, git, package files, AST analysis).

Components:
- FileInventory: Tracks important files by category
- DependencyGraph: Maps module dependencies
- SymbolIndex: Quick lookup of functions/classes/exports
- ArchitectureMap: High-level architecture overview

Usage:
    manifest = await ProjectManifest.build("/path/to/project")

    # Get context for a specific task
    context = manifest.get_context_for_task("implement authentication")

    # Get important files for a query
    relevant = manifest.get_relevant_files("user login")
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FileCategory(Enum):
    """Categories of files in a project."""

    ENTRY_POINT = "entry_point"  # Main entry files
    CONFIG = "config"  # Configuration files
    MODEL = "model"  # Data models/schemas
    API = "api"  # API routes/endpoints
    SERVICE = "service"  # Business logic services
    UTILITY = "utility"  # Utility/helper modules
    TEST = "test"  # Test files
    DOCUMENTATION = "documentation"  # Docs and READMEs
    BUILD = "build"  # Build/deploy scripts
    STYLE = "style"  # CSS/styling files
    ASSET = "asset"  # Static assets
    DATABASE = "database"  # Database migrations/schemas
    UNKNOWN = "unknown"


@dataclass
class FileInfo:
    """Information about a project file.

    Attributes:
        path: Relative path from project root
        category: File category
        language: Programming language
        size_bytes: File size
        line_count: Number of lines
        imports: List of imports/dependencies
        exports: List of exported symbols
        description: Brief description (from docstring/comments)
        importance: Importance score (0.0 to 1.0)
        last_modified: Last modification timestamp
    """

    path: str
    category: FileCategory = FileCategory.UNKNOWN
    language: str = ""
    size_bytes: int = 0
    line_count: int = 0
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    description: str = ""
    importance: float = 0.5
    last_modified: float = 0.0


@dataclass
class ModuleInfo:
    """Information about a code module/package.

    Attributes:
        name: Module name
        path: Path to module directory or file
        files: Files in this module
        dependencies: Other modules this depends on
        dependents: Modules that depend on this
        public_api: Public functions/classes
        description: Module description
    """

    name: str
    path: str
    files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    public_api: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ArchitectureLayer:
    """A layer in the architecture.

    Attributes:
        name: Layer name (e.g., "presentation", "business", "data")
        modules: Modules in this layer
        description: Layer description
        allowed_dependencies: Layers this can depend on
    """

    name: str
    modules: List[str] = field(default_factory=list)
    description: str = ""
    allowed_dependencies: List[str] = field(default_factory=list)


@dataclass
class ProjectMetadata:
    """Project metadata from package files.

    Attributes:
        name: Project name
        version: Project version
        description: Project description
        language: Primary language
        framework: Primary framework (e.g., Django, React)
        package_manager: Package manager (npm, pip, cargo)
        dependencies: External dependencies
        dev_dependencies: Development dependencies
        scripts: Available scripts/commands
    """

    name: str = ""
    version: str = ""
    description: str = ""
    language: str = ""
    framework: str = ""
    package_manager: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    dev_dependencies: Dict[str, str] = field(default_factory=dict)
    scripts: Dict[str, str] = field(default_factory=dict)


class ProjectManifest:
    """Structured representation of a project's codebase.

    Provides organized access to project files, modules, and architecture
    for better LLM context.
    """

    # File patterns for categorization
    CATEGORY_PATTERNS = {
        FileCategory.ENTRY_POINT: [
            "main.py",
            "app.py",
            "__main__.py",
            "index.js",
            "index.ts",
            "main.go",
            "main.rs",
            "Main.java",
        ],
        FileCategory.CONFIG: [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.toml",
            "*.ini",
            "*.cfg",
            ".env*",
            "Makefile",
            "Dockerfile",
            "docker-compose*",
        ],
        FileCategory.TEST: [
            "test_*.py",
            "*_test.py",
            "*.test.js",
            "*.test.ts",
            "*.spec.js",
            "*.spec.ts",
            "*Test.java",
            "*_test.go",
        ],
        FileCategory.DOCUMENTATION: [
            "*.md",
            "*.rst",
            "*.txt",
            "LICENSE*",
            "CHANGELOG*",
        ],
        FileCategory.BUILD: [
            "setup.py",
            "setup.cfg",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "pom.xml",
            "build.gradle",
        ],
        FileCategory.STYLE: [
            "*.css",
            "*.scss",
            "*.sass",
            "*.less",
        ],
        FileCategory.DATABASE: [
            "*migration*",
            "*schema*",
            "*.sql",
        ],
    }

    # Language detection by extension
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
    }

    # Directories to skip
    SKIP_DIRS = {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "dist",
        "build",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        "target",
        ".idea",
        ".vscode",
        "coverage",
        ".next",
    }

    def __init__(self, project_root: str):
        """Initialize the manifest.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.files: Dict[str, FileInfo] = {}
        self.modules: Dict[str, ModuleInfo] = {}
        self.metadata = ProjectMetadata()
        self.architecture: List[ArchitectureLayer] = []
        self._file_index: Dict[str, Set[str]] = defaultdict(set)
        self._symbol_index: Dict[str, str] = {}

    @classmethod
    async def build(
        cls,
        project_root: str,
        include_analysis: bool = True,
        max_files: int = 1000,
    ) -> "ProjectManifest":
        """Build a project manifest from a directory.

        Args:
            project_root: Root directory of the project
            include_analysis: Whether to analyze file contents
            max_files: Maximum files to scan

        Returns:
            Populated ProjectManifest
        """
        manifest = cls(project_root)

        # Scan project structure
        await manifest._scan_files(max_files)

        # Load metadata from package files
        await manifest._load_metadata()

        # Analyze file contents if requested
        if include_analysis:
            await manifest._analyze_files()

        # Build module graph
        await manifest._build_module_graph()

        # Infer architecture
        await manifest._infer_architecture()

        logger.info(
            f"Built manifest: {len(manifest.files)} files, " f"{len(manifest.modules)} modules"
        )

        return manifest

    async def _scan_files(self, max_files: int) -> None:
        """Scan project files and categorize them."""
        file_count = 0

        for root, dirs, files in os.walk(self.project_root):
            # Filter out skipped directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]

            for filename in files:
                if file_count >= max_files:
                    return

                full_path = Path(root) / filename
                rel_path = str(full_path.relative_to(self.project_root))

                # Skip hidden files
                if filename.startswith("."):
                    continue

                try:
                    stat = full_path.stat()
                    file_info = FileInfo(
                        path=rel_path,
                        size_bytes=stat.st_size,
                        last_modified=stat.st_mtime,
                    )

                    # Detect language
                    ext = full_path.suffix.lower()
                    file_info.language = self.LANGUAGE_MAP.get(ext, "")

                    # Categorize
                    file_info.category = self._categorize_file(rel_path, filename)

                    # Calculate importance
                    file_info.importance = self._calculate_importance(file_info)

                    self.files[rel_path] = file_info
                    self._file_index[file_info.category.value].add(rel_path)
                    file_count += 1

                except Exception as e:
                    logger.debug(f"Error scanning {rel_path}: {e}")

    def _categorize_file(self, path: str, filename: str) -> FileCategory:
        """Categorize a file based on its name and path."""
        import fnmatch

        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return category

        # Path-based categorization
        path_lower = path.lower()

        if "/test" in path_lower or "tests/" in path_lower:
            return FileCategory.TEST

        if "/api/" in path_lower or "/routes/" in path_lower:
            return FileCategory.API

        if "/models/" in path_lower or "/schemas/" in path_lower:
            return FileCategory.MODEL

        if "/services/" in path_lower or "/service/" in path_lower:
            return FileCategory.SERVICE

        if "/utils/" in path_lower or "/helpers/" in path_lower:
            return FileCategory.UTILITY

        if "/static/" in path_lower or "/assets/" in path_lower:
            return FileCategory.ASSET

        return FileCategory.UNKNOWN

    def _calculate_importance(self, file_info: FileInfo) -> float:
        """Calculate importance score for a file."""
        score = 0.5  # Base score

        # Entry points are most important
        if file_info.category == FileCategory.ENTRY_POINT:
            score = 0.95

        # Config and API files are important
        elif file_info.category in (FileCategory.CONFIG, FileCategory.API):
            score = 0.8

        # Models and services
        elif file_info.category in (FileCategory.MODEL, FileCategory.SERVICE):
            score = 0.7

        # Tests are moderately important
        elif file_info.category == FileCategory.TEST:
            score = 0.6

        # Documentation and assets are lower priority
        elif file_info.category in (FileCategory.DOCUMENTATION, FileCategory.ASSET):
            score = 0.4

        # Adjust by path depth (shallower = more important)
        depth = file_info.path.count("/")
        score -= min(depth * 0.05, 0.2)

        return max(0.1, min(1.0, score))

    async def _load_metadata(self) -> None:
        """Load project metadata from package files."""
        # Check for Python project
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            await self._load_pyproject_toml(pyproject)
            return

        setup_py = self.project_root / "setup.py"
        if setup_py.exists():
            self.metadata.language = "python"
            self.metadata.package_manager = "pip"

        # Check for Node.js project
        package_json = self.project_root / "package.json"
        if package_json.exists():
            await self._load_package_json(package_json)
            return

        # Check for Go project
        go_mod = self.project_root / "go.mod"
        if go_mod.exists():
            self.metadata.language = "go"
            self.metadata.package_manager = "go"

        # Check for Rust project
        cargo_toml = self.project_root / "Cargo.toml"
        if cargo_toml.exists():
            self.metadata.language = "rust"
            self.metadata.package_manager = "cargo"

    async def _load_pyproject_toml(self, path: Path) -> None:
        """Load metadata from pyproject.toml."""
        try:
            import tomli
        except ImportError:
            try:
                import tomllib as tomli
            except ImportError:
                return

        try:
            content = path.read_text()
            data = tomli.loads(content)

            project = data.get("project", {})
            self.metadata.name = project.get("name", "")
            self.metadata.version = project.get("version", "")
            self.metadata.description = project.get("description", "")
            self.metadata.language = "python"
            self.metadata.package_manager = "pip"

            deps = project.get("dependencies", [])
            self.metadata.dependencies = {d.split(">=")[0].split("==")[0]: "" for d in deps}

            # Detect framework
            if "django" in self.metadata.dependencies:
                self.metadata.framework = "django"
            elif "flask" in self.metadata.dependencies:
                self.metadata.framework = "flask"
            elif "fastapi" in self.metadata.dependencies:
                self.metadata.framework = "fastapi"

        except Exception as e:
            logger.debug(f"Error loading pyproject.toml: {e}")

    async def _load_package_json(self, path: Path) -> None:
        """Load metadata from package.json."""
        try:
            content = path.read_text()
            data = json.loads(content)

            self.metadata.name = data.get("name", "")
            self.metadata.version = data.get("version", "")
            self.metadata.description = data.get("description", "")
            self.metadata.language = "javascript"
            self.metadata.package_manager = "npm"
            self.metadata.dependencies = data.get("dependencies", {})
            self.metadata.dev_dependencies = data.get("devDependencies", {})
            self.metadata.scripts = data.get("scripts", {})

            # Detect framework
            deps = {**self.metadata.dependencies, **self.metadata.dev_dependencies}
            if "react" in deps:
                self.metadata.framework = "react"
            elif "vue" in deps:
                self.metadata.framework = "vue"
            elif "angular" in deps:
                self.metadata.framework = "angular"
            elif "express" in deps:
                self.metadata.framework = "express"
            elif "next" in deps:
                self.metadata.framework = "next.js"

        except Exception as e:
            logger.debug(f"Error loading package.json: {e}")

    async def _analyze_files(self) -> None:
        """Analyze file contents for imports/exports."""
        for path, file_info in self.files.items():
            if file_info.language not in ("python", "javascript", "typescript"):
                continue

            full_path = self.project_root / path
            if not full_path.exists():
                continue

            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                file_info.line_count = len(lines)

                if file_info.language == "python":
                    self._analyze_python_file(content, file_info)
                elif file_info.language in ("javascript", "typescript"):
                    self._analyze_js_file(content, file_info)

                # Extract description from docstring/comments
                file_info.description = self._extract_description(content, file_info.language)

            except Exception as e:
                logger.debug(f"Error analyzing {path}: {e}")

    def _analyze_python_file(self, content: str, file_info: FileInfo) -> None:
        """Extract imports and exports from Python file."""
        import re

        # Find imports
        import_pattern = re.compile(r"^(?:from\s+([\w.]+)\s+)?import\s+([\w,\s]+)", re.MULTILINE)
        for match in import_pattern.finditer(content):
            module = match.group(1) or match.group(2).split(",")[0].strip()
            file_info.imports.append(module)

        # Find exports (class and function definitions)
        export_pattern = re.compile(r"^(?:class|def)\s+(\w+)", re.MULTILINE)
        for match in export_pattern.finditer(content):
            symbol = match.group(1)
            if not symbol.startswith("_"):
                file_info.exports.append(symbol)
                self._symbol_index[symbol] = file_info.path

    def _analyze_js_file(self, content: str, file_info: FileInfo) -> None:
        """Extract imports and exports from JavaScript/TypeScript file."""
        import re

        # Find imports
        import_pattern = re.compile(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
        for match in import_pattern.finditer(content):
            file_info.imports.append(match.group(1))

        # Find exports
        export_pattern = re.compile(
            r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)",
            re.MULTILINE,
        )
        for match in export_pattern.finditer(content):
            symbol = match.group(1)
            file_info.exports.append(symbol)
            self._symbol_index[symbol] = file_info.path

    def _extract_description(self, content: str, language: str) -> str:
        """Extract description from file docstring/comments."""
        lines = content.split("\n")[:20]  # Only check first 20 lines

        if language == "python":
            # Look for module docstring
            in_docstring = False
            docstring = []
            for line in lines:
                if '"""' in line or "'''" in line:
                    if in_docstring:
                        break
                    in_docstring = True
                    # Get text after opening quotes
                    idx = max(line.find('"""'), line.find("'''"))
                    text = line[idx + 3 :].strip()
                    if text:
                        docstring.append(text)
                elif in_docstring:
                    docstring.append(line.strip())

            if docstring:
                return " ".join(docstring[:2])  # First 2 lines

        # Look for header comments
        comment_char = "//" if language in ("javascript", "typescript") else "#"
        comments = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(comment_char):
                text = stripped.lstrip(comment_char).strip()
                if text and not text.startswith("!"):  # Skip shebang-style comments
                    comments.append(text)
            elif stripped and not stripped.startswith(comment_char):
                break

        if comments:
            return " ".join(comments[:2])

        return ""

    async def _build_module_graph(self) -> None:
        """Build module dependency graph."""
        # Group files by module (directory)
        module_files: Dict[str, List[str]] = defaultdict(list)

        for path in self.files:
            parts = path.split("/")
            if len(parts) > 1:
                module_name = parts[0]
            else:
                module_name = "_root"

            module_files[module_name].append(path)

        # Create module info
        for module_name, files in module_files.items():
            module_info = ModuleInfo(
                name=module_name,
                path=module_name if module_name != "_root" else "",
                files=files,
            )

            # Aggregate imports and exports
            all_imports: Set[str] = set()
            all_exports: Set[str] = set()

            for file_path in files:
                file_info = self.files.get(file_path)
                if file_info:
                    all_imports.update(file_info.imports)
                    all_exports.update(file_info.exports)

            module_info.public_api = list(all_exports)

            # Determine dependencies (imports that reference other modules)
            for imp in all_imports:
                dep_module = imp.split(".")[0]
                if dep_module in module_files and dep_module != module_name:
                    module_info.dependencies.append(dep_module)

            self.modules[module_name] = module_info

        # Build reverse dependencies
        for module_name, module_info in self.modules.items():
            for dep in module_info.dependencies:
                if dep in self.modules:
                    self.modules[dep].dependents.append(module_name)

    async def _infer_architecture(self) -> None:
        """Infer architectural layers from project structure."""
        # Common layer patterns
        layer_keywords = {
            "presentation": ["ui", "view", "component", "page", "screen", "template"],
            "api": ["api", "route", "endpoint", "controller", "handler"],
            "business": ["service", "logic", "domain", "use_case", "usecase"],
            "data": ["model", "schema", "database", "repository", "dao", "orm"],
            "infrastructure": ["infra", "config", "util", "helper", "lib"],
        }

        layers: Dict[str, List[str]] = defaultdict(list)

        for module_name in self.modules:
            module_lower = module_name.lower()
            for layer_name, keywords in layer_keywords.items():
                if any(kw in module_lower for kw in keywords):
                    layers[layer_name].append(module_name)
                    break
            else:
                layers["other"].append(module_name)

        # Create architecture layers
        self.architecture = [
            ArchitectureLayer(
                name="presentation",
                modules=layers.get("presentation", []),
                description="User interface components",
                allowed_dependencies=["business", "infrastructure"],
            ),
            ArchitectureLayer(
                name="api",
                modules=layers.get("api", []),
                description="API endpoints and controllers",
                allowed_dependencies=["business", "infrastructure"],
            ),
            ArchitectureLayer(
                name="business",
                modules=layers.get("business", []),
                description="Business logic and domain services",
                allowed_dependencies=["data", "infrastructure"],
            ),
            ArchitectureLayer(
                name="data",
                modules=layers.get("data", []),
                description="Data access and persistence",
                allowed_dependencies=["infrastructure"],
            ),
            ArchitectureLayer(
                name="infrastructure",
                modules=layers.get("infrastructure", []),
                description="Cross-cutting concerns and utilities",
                allowed_dependencies=[],
            ),
        ]

    def get_important_files(self, limit: int = 20) -> List[FileInfo]:
        """Get most important files in the project.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of FileInfo sorted by importance
        """
        sorted_files = sorted(
            self.files.values(),
            key=lambda f: f.importance,
            reverse=True,
        )
        return sorted_files[:limit]

    def get_files_by_category(self, category: FileCategory) -> List[FileInfo]:
        """Get files by category.

        Args:
            category: File category

        Returns:
            List of FileInfo in that category
        """
        paths = self._file_index.get(category.value, set())
        return [self.files[p] for p in paths if p in self.files]

    def get_relevant_files(self, query: str, limit: int = 10) -> List[FileInfo]:
        """Get files relevant to a query.

        Args:
            query: Search query
            limit: Maximum files to return

        Returns:
            List of relevant FileInfo
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored_files: List[tuple] = []

        for path, file_info in self.files.items():
            score = file_info.importance

            # Boost if path matches query
            path_lower = path.lower()
            for term in query_terms:
                if term in path_lower:
                    score += 0.3

            # Boost if description matches
            if file_info.description:
                desc_lower = file_info.description.lower()
                for term in query_terms:
                    if term in desc_lower:
                        score += 0.2

            # Boost if exports match
            for export in file_info.exports:
                export_lower = export.lower()
                for term in query_terms:
                    if term in export_lower:
                        score += 0.25

            scored_files.append((score, file_info))

        # Sort by score descending
        scored_files.sort(key=lambda x: x[0], reverse=True)

        return [f[1] for f in scored_files[:limit]]

    def find_symbol(self, symbol: str) -> Optional[str]:
        """Find which file defines a symbol.

        Args:
            symbol: Symbol name (function, class, etc.)

        Returns:
            File path or None
        """
        return self._symbol_index.get(symbol)

    def get_context_for_task(self, task: str) -> Dict[str, Any]:
        """Generate context for a specific task.

        Args:
            task: Task description

        Returns:
            Context dictionary with relevant project info
        """
        relevant_files = self.get_relevant_files(task, limit=15)

        return {
            "project": {
                "name": self.metadata.name,
                "language": self.metadata.language,
                "framework": self.metadata.framework,
                "description": self.metadata.description,
            },
            "relevant_files": [
                {
                    "path": f.path,
                    "category": f.category.value,
                    "description": f.description,
                    "exports": f.exports[:5],
                }
                for f in relevant_files
            ],
            "architecture": [
                {
                    "layer": layer.name,
                    "modules": layer.modules,
                    "description": layer.description,
                }
                for layer in self.architecture
                if layer.modules
            ],
            "entry_points": [f.path for f in self.get_files_by_category(FileCategory.ENTRY_POINT)],
            "dependencies": list(self.metadata.dependencies.keys())[:20],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export manifest to dictionary."""
        return {
            "metadata": {
                "name": self.metadata.name,
                "version": self.metadata.version,
                "description": self.metadata.description,
                "language": self.metadata.language,
                "framework": self.metadata.framework,
            },
            "file_count": len(self.files),
            "module_count": len(self.modules),
            "categories": {
                cat.value: len(self._file_index.get(cat.value, set())) for cat in FileCategory
            },
            "top_files": [
                {"path": f.path, "importance": f.importance} for f in self.get_important_files(10)
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ProjectManifest({self.metadata.name or self.project_root}, "
            f"files={len(self.files)}, modules={len(self.modules)})"
        )
