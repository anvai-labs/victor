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

"""Smart codebase analyzer for generating comprehensive init.md files.

This module analyzes Python codebases to extract:
- Package structure and layout
- Key classes and their locations (with line numbers)
- Architectural patterns (providers, tools, managers, etc.)
- CLI commands from pyproject.toml
- Configuration files and their purposes

Output location: .victor/init.md (configurable via settings.py)
"""

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Lazy load ignore patterns from external victor-coding package
try:
    from victor_coding.codebase.ignore_patterns import (
        DEFAULT_SKIP_DIRS,
        is_hidden_path,
        should_ignore_path,
    )

    _IGNORE_PATTERNS_AVAILABLE = True
except ImportError:
    _IGNORE_PATTERNS_AVAILABLE = False
    DEFAULT_SKIP_DIRS = frozenset(
        {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "*.pyc",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            "*.egg-info",
        }
    )

    def is_hidden_path(path: Path) -> bool:
        return path.name.startswith(".")

    def should_ignore_path(
        path: Path, skip_dirs: frozenset, extra_skip_dirs: Optional[frozenset] = None
    ) -> bool:
        """Fallback implementation when victor-coding is not available."""
        # Simple fallback: check if path is hidden or in skip_dirs
        if path.name.startswith("."):
            return True
        if extra_skip_dirs and any(part in extra_skip_dirs for part in path.parts):
            return True
        return any(skip_dir in path.parts for skip_dir in skip_dirs)


from victor_coding.compat.settings import VICTOR_CONTEXT_FILE, get_project_paths
from victor_contracts.utils.ast_helpers import (
    extract_base_classes,
    extract_imports,
    is_stdlib_module,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassInfo:
    """Information about a discovered class."""

    name: str
    file_path: str
    line_number: int
    base_classes: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False
    category: Optional[str] = None  # e.g., "provider", "tool", "manager"


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    name: str
    path: str
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class CodebaseAnalysis:
    """Complete analysis of a codebase."""

    project_name: str
    root_path: Path
    main_package: Optional[str] = None
    deprecated_paths: List[str] = field(default_factory=list)
    packages: Dict[str, List[ModuleInfo]] = field(default_factory=dict)
    key_components: List[ClassInfo] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    cli_commands: List[str] = field(default_factory=list)
    architecture_patterns: List[str] = field(default_factory=list)
    config_files: List[Tuple[str, str]] = field(default_factory=list)  # (path, description)
    # Enhanced fields
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # {category: [dep1, dep2]}
    test_coverage: Optional[float] = None  # Coverage percentage if available
    loc_stats: Dict[str, int] = field(
        default_factory=dict
    )  # {total_lines, total_files, largest_file}
    top_imports: List[Tuple[str, int]] = field(default_factory=list)  # [(module, import_count)]
    method_count: int = 0
    protocol_count: int = 0  # Python Protocol/ABC count


class CodebaseAnalyzer:
    """Analyzes codebases to extract structure and architecture (language-agnostic).

    Decomposition targets (SRP):
    - SymbolExtractorMixin: _analyze_source_files through _categorize_class (lines 304-532)
    - ArchitectureAnalyzerMixin: _identify_key_components through _extract_test_coverage (lines 534-950)
    - CodebaseAnalyzer: __init__, analyze() facade, constants, _detect_package_layout
    See victor_coding/codebase/symbol_extractor.py and architecture_analyzer.py.
    """

    # Patterns that indicate key architectural components (universal across languages)
    KEY_CLASS_PATTERNS = {
        "provider": ["Provider", "Backend", "Client", "Connector", "Adapter"],
        "tool": ["Tool", "Command", "Action", "Handler", "Executor"],
        "manager": ["Manager", "Orchestrator", "Controller", "Coordinator", "Director"],
        "model": ["Model", "Schema", "Entity", "Record", "DTO", "ViewModel"],
        "config": ["Config", "Settings", "Options", "Preferences", "Environment"],
        "base": [
            "Base",
            "Abstract",
            "Interface",
            "I[A-Z]",
        ],  # IService, IRepository etc.
        "registry": ["Registry", "Repository", "Store", "Cache", "Factory"],
        "service": ["Service", "Worker", "Processor", "UseCase", "Interactor"],
        "component": ["Component", "Widget", "View", "Screen", "Page"],
        "middleware": ["Middleware", "Interceptor", "Filter", "Guard"],
        "router": ["Router", "Route", "Endpoint", "Controller"],
    }

    # Source file extensions by language
    LANGUAGE_EXTENSIONS = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript React",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".java": "Java",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".cs": "C#",
        ".cpp": "C++",
        ".c": "C",
        ".swift": "Swift",
        ".dart": "Dart",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }

    # Config and documentation file extensions (counted for LOC but not parsed)
    CONFIG_EXTENSIONS = {
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".ini": "INI",
        ".hocon": "HOCON",
        ".xml": "XML",
        ".md": "Markdown",
        ".txt": "Text",
        ".cfg": "Config",
        ".conf": "Config",
        ".props": "Properties",
    }

    # Use shared default skip directories from ignore_patterns module
    # Hidden directories (starting with '.') are excluded automatically
    # by the shared should_ignore_path() utility
    SKIP_DIRS = DEFAULT_SKIP_DIRS

    def __init__(
        self,
        root_path: Optional[str] = None,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        """Initialize analyzer.

        Args:
            root_path: Root directory to analyze. Defaults to current directory.
            include_dirs: List of directories to include in the analysis.
            exclude_dirs: List of directories to exclude from the analysis.
        """
        self.root = Path(root_path).resolve() if root_path else Path.cwd()
        self.analysis = CodebaseAnalysis(project_name=self.root.name, root_path=self.root)
        self.include_dirs = include_dirs

        # Combine default and user-provided exclude dirs
        self.effective_skip_dirs = self.SKIP_DIRS.copy()
        if exclude_dirs:
            self.effective_skip_dirs.update(exclude_dirs)

    def analyze(self) -> CodebaseAnalysis:
        """Perform full codebase analysis (language-agnostic).

        Delegates to SymbolExtractor and ArchitectureAnalyzer for SRP.

        Returns:
            Complete CodebaseAnalysis object.
        """
        from victor_coding.codebase.symbol_extractor import SymbolExtractor
        from victor_coding.codebase.architecture_analyzer import ArchitectureAnalyzer

        logger.info(f"Analyzing codebase at {self.root}")

        # Step 1: Detect package/source layout (stays here — lightweight)
        self._detect_package_layout()

        # Step 2: Symbol extraction (delegated)
        extractor = SymbolExtractor(
            root=self.root,
            analysis=self.analysis,
            include_dirs=self.include_dirs,
            effective_skip_dirs=self.effective_skip_dirs,
            language_extensions=self.LANGUAGE_EXTENSIONS,
            key_class_patterns=self.KEY_CLASS_PATTERNS,
        )
        extractor.analyze_source_files()

        # Steps 3-10: Architecture analysis (delegated)
        arch = ArchitectureAnalyzer(
            root=self.root,
            analysis=self.analysis,
            include_dirs=self.include_dirs,
            effective_skip_dirs=self.effective_skip_dirs,
            language_extensions=self.LANGUAGE_EXTENSIONS,
            config_extensions=self.CONFIG_EXTENSIONS,
        )
        arch.identify_key_components()
        arch.extract_entry_points()
        arch.detect_architecture_patterns()
        arch.find_config_files()
        arch.extract_dependencies()
        arch.calculate_loc_stats()
        arch.extract_top_imports()
        arch.extract_test_coverage()

        return self.analysis

    def _detect_package_layout(self) -> None:
        """Detect the package/source layout (language-agnostic)."""

        def is_python_package(path: Path) -> bool:
            return path.is_dir() and (path / "__init__.py").exists()

        def is_source_directory(path: Path) -> bool:
            """Check if directory contains source files."""
            if not path.is_dir():
                return False
            for ext in self.LANGUAGE_EXTENSIONS.keys():
                if list(path.glob(f"*{ext}")):
                    return True
            return False

        # Find root-level packages/source directories
        source_dirs = []
        python_packages = []

        for item in self.root.iterdir():
            # Use shared ignore logic for consistency
            if should_ignore_path(item, skip_dirs=self.effective_skip_dirs):
                continue
            if is_python_package(item):
                python_packages.append(item.name)
            elif is_source_directory(item):
                source_dirs.append(item.name)

        has_src = (self.root / "src").is_dir()
        has_lib = (self.root / "lib").is_dir()
        has_app = (self.root / "app").is_dir()

        # Prioritize Python packages, then common source dirs
        if python_packages:
            main_candidates = [p for p in python_packages if not p.startswith("test")]
            self.analysis.main_package = (
                main_candidates[0] if main_candidates else python_packages[0]
            )
            if has_src and "src" not in python_packages:
                self.analysis.deprecated_paths.append("src/")
        elif has_src:
            # Check for packages inside src/
            src_packages = [d.name for d in (self.root / "src").iterdir() if is_python_package(d)]
            if src_packages:
                self.analysis.main_package = f"src/{src_packages[0]}"
            elif is_source_directory(self.root / "src"):
                self.analysis.main_package = "src"
        elif has_lib and is_source_directory(self.root / "lib"):
            self.analysis.main_package = "lib"
        elif has_app and is_source_directory(self.root / "app"):
            self.analysis.main_package = "app"
        elif source_dirs:
            # Use first source directory found
            main_candidates = [d for d in source_dirs if d not in ("tests", "test", "spec")]
            self.analysis.main_package = main_candidates[0] if main_candidates else source_dirs[0]

    def _analyze_source_files(self) -> None:
        """Analyze source files across all supported languages."""
        search_paths = []
        if self.include_dirs:
            for d in self.include_dirs:
                path = self.root / d
                if path.exists() and path.is_dir():
                    search_paths.append(path)

        if not search_paths:
            # Fallback to original logic if no include_dirs provided or none exist
            if self.analysis.main_package:
                main_path = self.root / self.analysis.main_package.replace("/", "/")
                if main_path.exists():
                    search_paths.append(main_path)
            if not search_paths:
                for common_dir in ["src", "lib", "app", "components", "pages", "api"]:
                    path = self.root / common_dir
                    if path.exists():
                        search_paths.append(path)
            if not search_paths:
                search_paths.append(self.root)

        for search_path in search_paths:
            self._scan_directory_for_sources(search_path)

    def _scan_directory_for_sources(self, directory: Path, max_depth: int = 5) -> None:
        """Scan directory for source files of any language."""
        for ext, lang in self.LANGUAGE_EXTENSIONS.items():
            for source_file in directory.rglob(f"*{ext}"):
                # Use shared ignore logic (handles hidden dirs and skip dirs)
                if should_ignore_path(source_file, skip_dirs=self.effective_skip_dirs):
                    continue

                # Check depth
                try:
                    rel_path = source_file.relative_to(self.root)
                    if len(rel_path.parts) > max_depth:
                        continue
                except ValueError:
                    continue

                # Parse based on language
                if ext == ".py":
                    module_info = self._parse_python_file(source_file, str(rel_path))
                else:
                    module_info = self._parse_generic_file(source_file, str(rel_path), lang)

                if module_info:
                    # Organize by subpackage/directory
                    parts = rel_path.parts
                    if len(parts) > 1:
                        subpackage = parts[0]
                    else:
                        subpackage = "root"

                    if subpackage not in self.analysis.packages:
                        self.analysis.packages[subpackage] = []
                    self.analysis.packages[subpackage].append(module_info)

    def _parse_generic_file(
        self, file_path: Path, rel_path: str, language: str
    ) -> Optional[ModuleInfo]:
        """Parse any source file using regex patterns to extract components.

        This is language-agnostic - detects class/interface/struct/function patterns.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return None

        module_info = ModuleInfo(name=file_path.stem, path=rel_path)

        # Universal patterns for class/component detection
        patterns = [
            # class ClassName, export class, public class, etc.
            r"(?:export\s+)?(?:public\s+|private\s+|abstract\s+)?class\s+([A-Z][a-zA-Z0-9_]*)",
            # interface IName or interface Name
            r"(?:export\s+)?interface\s+([A-Z][a-zA-Z0-9_]*)",
            # struct Name (Go, Rust, C)
            r"(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9_]*)",
            # type Name = (TypeScript type aliases)
            r"(?:export\s+)?type\s+([A-Z][a-zA-Z0-9_]*)\s*=",
            # enum Name
            r"(?:export\s+)?(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9_]*)",
            # trait Name (Rust)
            r"(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9_]*)",
            # module Name (Ruby, Elixir)
            r"(?:defmodule|module)\s+([A-Z][a-zA-Z0-9_:]*)",
            # React components: function ComponentName or const ComponentName =
            r"(?:export\s+)?(?:const|function)\s+([A-Z][a-zA-Z0-9_]*)\s*[=\(]",
            # Vue/Svelte component detection via filename
        ]

        for line_no, line in enumerate(content.split("\n"), 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    class_name = match.group(1)
                    # Get brief description from comment above or same line
                    desc = self._extract_inline_comment(line, content, line_no)
                    category = self._categorize_class(class_name, [])

                    class_info = ClassInfo(
                        name=class_name,
                        file_path=rel_path,
                        line_number=line_no,
                        base_classes=[],
                        docstring=desc,
                        is_abstract="abstract" in line.lower() or "interface" in line.lower(),
                        category=category,
                    )
                    module_info.classes.append(class_info)
                    break  # One match per line

        return module_info if module_info.classes else None

    def _extract_inline_comment(self, line: str, content: str, line_no: int) -> Optional[str]:
        """Extract comment from line or line above."""
        # Check for inline comment
        for comment_marker in ["//", "#", "--", "/*", "///"]:
            if comment_marker in line:
                idx = line.find(comment_marker)
                comment = line[idx + len(comment_marker) :].strip()
                if comment:
                    return comment[:60]

        # Check line above for doc comment
        lines = content.split("\n")
        if line_no > 1:
            prev_line = lines[line_no - 2].strip()
            for marker in ["///", "/**", "//", "#", '"""', "'''"]:
                if prev_line.startswith(marker):
                    return prev_line.lstrip(marker).strip("*/ ").strip()[:60]

        return None

    def _parse_python_file(self, file_path: Path, rel_path: str) -> Optional[ModuleInfo]:
        """Parse a Python file and extract class/function information.

        Args:
            file_path: Absolute path to the file
            rel_path: Relative path from project root

        Returns:
            ModuleInfo or None if parsing fails
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return None

        module_info = ModuleInfo(
            name=file_path.stem,
            path=rel_path,
        )

        # Get module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            module_info.description = tree.body[0].value.value

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, rel_path)
                module_info.classes.append(class_info)
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                # Top-level function
                module_info.functions.append(node.name)

        return module_info

    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract information from a class AST node."""
        # Get base class names (short names for categorization compatibility)
        base_classes = [b.rsplit(".", 1)[-1] for b in extract_base_classes(node)]

        # Get docstring
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            doc = node.body[0].value.value
            if isinstance(doc, str):
                # Get first line only
                docstring = doc.split("\n")[0].strip()

        # Check if abstract
        is_abstract = (
            any(
                isinstance(d, ast.Name) and d.id in ("abstractmethod", "ABC")
                for d in node.decorator_list
            )
            or "ABC" in base_classes
            or "Abstract" in node.name
        )

        # Categorize the class
        category = self._categorize_class(node.name, base_classes)

        return ClassInfo(
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            base_classes=base_classes,
            docstring=docstring,
            is_abstract=is_abstract,
            category=category,
        )

    def _categorize_class(self, name: str, base_classes: List[str]) -> Optional[str]:
        """Categorize a class based on its name and base classes."""
        all_names = [name] + base_classes

        for category, patterns in self.KEY_CLASS_PATTERNS.items():
            for pattern in patterns:
                if any(pattern in n for n in all_names):
                    return category
        return None

    def _identify_key_components(self) -> None:
        """Identify the most important classes in the codebase."""
        all_classes: List[ClassInfo] = []
        method_count = 0
        protocol_count = 0

        for modules in self.analysis.packages.values():
            for module in modules:
                all_classes.extend(module.classes)
                # Count methods (top-level functions are tracked separately)
                method_count += len(module.functions)
                # Count protocols/ABCs
                for cls in module.classes:
                    if (
                        cls.is_abstract
                        or "Protocol" in cls.name
                        or any(base in ("Protocol", "ABC") for base in cls.base_classes)
                    ):
                        protocol_count += 1

        self.analysis.method_count = method_count
        self.analysis.protocol_count = protocol_count

        # Priority scoring for key components
        def score_class(cls: ClassInfo) -> int:
            score = 0
            # Base/abstract classes are important
            if cls.is_abstract or "Base" in cls.name:
                score += 10
            # Categorized classes are more important
            if cls.category:
                score += 5
                if cls.category in ("manager", "provider", "registry"):
                    score += 5
            # Orchestrator is the main coordinator - highest priority
            if "Orchestrator" in cls.name:
                score += 15
            # Classes with docstrings are better documented
            if cls.docstring:
                score += 2
            return score

        # Sort by score and take top components
        scored = [(score_class(c), c) for c in all_classes if c.category]
        scored.sort(key=lambda x: -x[0])

        # Take top 20 most important (increased from 15)
        self.analysis.key_components = [c for _, c in scored[:20]]

    def _extract_entry_points(self) -> None:
        """Extract CLI entry points from pyproject.toml."""
        pyproject = self.root / "pyproject.toml"
        if not pyproject.exists():
            return

        try:
            content = pyproject.read_text(encoding="utf-8")

            # Parse [project.scripts] section
            scripts_match = re.search(r"\[project\.scripts\](.*?)(?=\[|\Z)", content, re.DOTALL)
            if scripts_match:
                scripts_section = scripts_match.group(1)
                for line in scripts_section.strip().split("\n"):
                    if "=" in line:
                        parts = line.split("=", 1)
                        cmd = parts[0].strip().strip('"')
                        target = parts[1].strip().strip('"')
                        self.analysis.entry_points[cmd] = target

            # Extract dev dependencies for CLI commands
            if "[project.optional-dependencies]" in content:
                # Common test/dev commands
                if "pytest" in content:
                    self.analysis.cli_commands.append("pytest")
                if "black" in content:
                    self.analysis.cli_commands.append("black .")
                if "ruff" in content:
                    self.analysis.cli_commands.append("ruff check .")
                if "mypy" in content:
                    self.analysis.cli_commands.append("mypy " + (self.analysis.main_package or "."))

        except Exception as e:
            logger.debug(f"Failed to parse pyproject.toml: {e}")

    def _detect_architecture_patterns(self) -> None:
        """Detect common architectural patterns in the codebase."""
        patterns = []

        # Check for provider pattern
        provider_classes = [c for c in self.analysis.key_components if c.category == "provider"]
        if len(provider_classes) >= 2:
            base = next((c for c in provider_classes if c.is_abstract or "Base" in c.name), None)
            if base:
                patterns.append(
                    f"Provider Pattern: Base class `{base.name}` ({base.file_path}:{base.line_number})"
                )

        # Check for tool/command pattern
        tool_classes = [c for c in self.analysis.key_components if c.category == "tool"]
        if len(tool_classes) >= 2:
            base = next((c for c in tool_classes if c.is_abstract or "Base" in c.name), None)
            if base:
                patterns.append(
                    f"Tool/Command Pattern: Base class `{base.name}` ({base.file_path}:{base.line_number})"
                )

        # Check for registry pattern
        registry_classes = [c for c in self.analysis.key_components if c.category == "registry"]
        if registry_classes:
            patterns.append(f"Registry Pattern: {len(registry_classes)} registries found")

        # Check for manager/orchestrator
        manager_classes = [c for c in self.analysis.key_components if c.category == "manager"]
        if manager_classes:
            main = manager_classes[0]
            patterns.append(
                f"Orchestrator/Manager: `{main.name}` ({main.file_path}:{main.line_number})"
            )

        # Check for config pattern
        config_classes = [c for c in self.analysis.key_components if c.category == "config"]
        if config_classes:
            patterns.append(f"Configuration: {len(config_classes)} config classes (Pydantic-style)")

        self.analysis.architecture_patterns = patterns

    def _find_config_files(self) -> None:
        """Find important configuration files."""
        config_patterns = [
            (
                "pyproject.toml",
                "Project configuration, dependencies, and build settings",
            ),
            ("setup.py", "Legacy Python package setup"),
            ("setup.cfg", "Package configuration"),
            (".env.example", "Environment variable template"),
            ("docker-compose.yml", "Docker service definitions"),
            ("Dockerfile", "Container build instructions"),
            ("Makefile", "Build automation"),
            (".github/workflows/*.yml", "GitHub Actions CI/CD"),
            ("requirements.txt", "Python dependencies"),
        ]

        for pattern, description in config_patterns:
            if "*" in pattern:
                # Glob pattern
                matches = list(self.root.glob(pattern))
                if matches:
                    self.analysis.config_files.append((pattern, description))
            else:
                if (self.root / pattern).exists():
                    self.analysis.config_files.append((pattern, description))

    def _extract_dependencies(self) -> None:
        """Extract dependencies from pyproject.toml or package.json."""
        # Try pyproject.toml first (Python)
        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text(encoding="utf-8")
                deps: Dict[str, List[str]] = {"core": [], "dev": [], "optional": []}

                # Parse [project.dependencies]
                deps_match = re.search(
                    r"\[project\].*?dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL
                )
                if deps_match:
                    deps_text = deps_match.group(1)
                    for line in deps_text.split(","):
                        dep = line.strip().strip("\"'")
                        if dep and not dep.startswith("#"):
                            # Extract package name (before version specifier)
                            pkg_name = re.split(r"[<>=~!]", dep)[0].strip()
                            if pkg_name:
                                deps["core"].append(pkg_name)

                # Parse [project.optional-dependencies]
                opt_deps_match = re.search(
                    r"\[project\.optional-dependencies\](.*?)(?=\[|\Z)",
                    content,
                    re.DOTALL,
                )
                if opt_deps_match:
                    opt_section = opt_deps_match.group(1)
                    # Extract dev dependencies
                    dev_match = re.search(r"dev\s*=\s*\[(.*?)\]", opt_section, re.DOTALL)
                    if dev_match:
                        for line in dev_match.group(1).split(","):
                            dep = line.strip().strip("\"'")
                            if dep and not dep.startswith("#"):
                                pkg_name = re.split(r"[<>=~!]", dep)[0].strip()
                                if pkg_name:
                                    deps["dev"].append(pkg_name)

                self.analysis.dependencies = deps

            except Exception as e:
                logger.debug(f"Failed to parse pyproject.toml dependencies: {e}")

        # Try package.json (Node.js)
        package_json = self.root / "package.json"
        if package_json.exists() and not self.analysis.dependencies:
            try:
                import json

                data = json.loads(package_json.read_text(encoding="utf-8"))
                deps: Dict[str, List[str]] = {"core": [], "dev": []}

                if "dependencies" in data:
                    deps["core"] = list(data["dependencies"].keys())[:15]
                if "devDependencies" in data:
                    deps["dev"] = list(data["devDependencies"].keys())[:10]

                self.analysis.dependencies = deps

            except Exception as e:
                logger.debug(f"Failed to parse package.json dependencies: {e}")

    def _calculate_loc_stats(self) -> None:
        """Calculate lines of code statistics for both source and config files."""
        # Source code stats
        source_lines = 0
        source_files = 0
        # Config/doc stats
        config_lines = 0
        config_files = 0

        # Overall tracking
        total_lines = 0
        total_files = 0
        largest_file = ""
        largest_file_lines = 0
        file_sizes: List[Tuple[str, int]] = []

        # Merge both extension types for unified scanning
        all_extensions = {**self.LANGUAGE_EXTENSIONS, **self.CONFIG_EXTENSIONS}

        # Scan all files
        search_dirs = (
            [self.root / d for d in self.include_dirs] if self.include_dirs else [self.root]
        )

        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for ext in all_extensions.keys():
                for source_file in search_dir.rglob(f"*{ext}"):
                    if should_ignore_path(source_file, skip_dirs=self.effective_skip_dirs):
                        continue
                    try:
                        content = source_file.read_text(encoding="utf-8", errors="ignore")
                        lines = len(content.splitlines())

                        # Track separately
                        is_config = ext in self.CONFIG_EXTENSIONS
                        if is_config:
                            config_lines += lines
                            config_files += 1
                        else:
                            source_lines += lines
                            source_files += 1

                        # Overall totals
                        total_lines += lines
                        total_files += 1
                        rel_path = str(source_file.relative_to(self.root))
                        file_sizes.append((rel_path, lines))
                        if lines > largest_file_lines:
                            largest_file_lines = lines
                            largest_file = rel_path
                    except Exception:
                        pass

        # Sort by size and get top 5
        file_sizes.sort(key=lambda x: -x[1])
        top_files = file_sizes[:5]

        self.analysis.loc_stats = {
            # Overall totals
            "total_lines": total_lines,
            "total_files": total_files,
            "largest_file": largest_file,
            "largest_file_lines": largest_file_lines,
            "top_files": top_files,
            # Breakdown by type
            "source_lines": source_lines,
            "source_files": source_files,
            "config_lines": config_lines,
            "config_files": config_files,
        }

    def _extract_top_imports(self) -> None:
        """Extract the most commonly imported modules (Python only).

        Uses AST parsing as primary method with regex fallback for notebooks
        and files with syntax errors (e.g., Databricks notebooks with magic commands).
        """
        import_counts: Dict[str, int] = defaultdict(int)

        # Only scan Python files
        search_dirs = (
            [self.root / d for d in self.include_dirs] if self.include_dirs else [self.root]
        )
        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for source_file in search_dir.rglob("*.py"):
                if should_ignore_path(source_file, skip_dirs=self.effective_skip_dirs):
                    continue
                try:
                    content = source_file.read_text(encoding="utf-8", errors="ignore")

                    # Try AST parsing first (fast, accurate)
                    try:
                        tree = ast.parse(content)
                        for module in extract_imports(tree, top_level_only=True):
                            import_counts[module] += 1
                    except SyntaxError:
                        # Fallback to regex for notebooks, Databricks files, etc.
                        for module in self._extract_imports_regex(content):
                            import_counts[module] += 1
                except Exception:
                    pass

        # Filter and sort (exclude stdlib/common third-party)
        filtered = [
            (mod, count) for mod, count in import_counts.items() if not is_stdlib_module(mod)
        ]
        filtered.sort(key=lambda x: -x[1])

        self.analysis.top_imports = filtered[:10]

    def _extract_imports_regex(self, content: str) -> List[str]:
        """Extract imports using regex as fallback for AST failures.

        Handles:
        - Regular Python imports
        - Databricks %run magic commands
        - Dynamic imports

        Args:
            content: File content to parse

        Returns:
            List of imported module names
        """
        imports = []

        # Regular Python imports
        # import module
        imports.extend(re.findall(r"^import\s+(\S+)", content, re.MULTILINE))
        # from module import ...
        imports.extend(re.findall(r"^from\s+(\S+)\s+import", content, re.MULTILINE))

        # Databricks-specific patterns
        # %run /path/to/notebook or %run ../path/to/module
        run_matches = re.findall(r"%run\s+(\S+)", content)
        for match in run_matches:
            # Extract module name from path
            # e.g., "../include/library/a.utils/import" -> "include.library.a.utils.import"
            parts = Path(match).parts
            if parts and parts[-1].endswith(".py"):
                # Remove .py extension
                parts = parts[:-1]
            if parts:
                imports.append(".".join(parts))

        # Dynamic imports
        # importlib.import_module("module.name")
        dynamic_imports = re.findall(r'importlib\.import_module\([\'"]([^\'"]+)[\'"]\)', content)
        imports.extend(dynamic_imports)

        # __import__("module.name")
        dynamic_imports2 = re.findall(r'__import__\([\'"]([^\'"]+)[\'"]\)', content)
        imports.extend(dynamic_imports2)

        return imports

    def _extract_test_coverage(self) -> None:
        """Try to extract test coverage from coverage reports."""
        # Check for coverage.xml
        coverage_file = self.root / "coverage.xml"
        if coverage_file.exists():
            try:
                content = coverage_file.read_text(encoding="utf-8")
                # Parse line-rate attribute (0.0 to 1.0)
                match = re.search(r'line-rate="([0-9.]+)"', content)
                if match:
                    coverage_pct = float(match.group(1)) * 100
                    self.analysis.test_coverage = round(coverage_pct, 1)
                    return
            except Exception:
                pass

        # Check for .coverage SQLite database
        coverage_db = self.root / ".coverage"
        if coverage_db.exists():
            try:
                import sqlite3

                conn = sqlite3.connect(str(coverage_db))
                cursor = conn.cursor()
                # Get line counts
                cursor.execute(
                    "SELECT SUM(num_statements), SUM(num_statements - missing_lines) FROM file_summary"
                )
                row = cursor.fetchone()
                if row and row[0]:
                    total, covered = row[0], row[1] or 0
                    self.analysis.test_coverage = round((covered / total) * 100, 1)
                conn.close()
            except Exception:
                pass

        # Check htmlcov/index.html for coverage percentage
        htmlcov = self.root / "htmlcov" / "index.html"
        if htmlcov.exists():
            try:
                content = htmlcov.read_text(encoding="utf-8")
                # Look for coverage percentage in the HTML
                match = re.search(r"(\d+)%\s*</span>\s*</h1>", content)
                if match:
                    self.analysis.test_coverage = float(match.group(1))
            except Exception:
                pass


def generate_smart_victor_md(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate comprehensive project context using codebase analysis.

    Works with Python projects (AST-based analysis) and falls back to
    language-agnostic analysis for non-Python projects.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Generated markdown content for .victor/init.md.
    """
    analyzer = CodebaseAnalyzer(root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs)
    analysis = analyzer.analyze()

    # If no Python package found, use language-agnostic analysis
    if not analysis.main_package and not analysis.key_components:
        return _generate_generic_victor_md(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )

    sections = []

    # Header - use configurable file name
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    sections.append("## Project Overview\n")

    # Try to get description from README
    readme_desc = _extract_readme_description(analysis.root_path)
    if readme_desc:
        sections.append(f"**{analysis.project_name}**: {readme_desc}\n")
    else:
        sections.append(f"**{analysis.project_name}**: [Add project description here]\n")

    # Package Layout
    sections.append("## Package Layout\n")
    sections.append("**IMPORTANT**: Use the correct directory paths:\n")

    layout_lines = []
    layout_lines.append("| Path | Status | Description |")
    layout_lines.append("|------|--------|-------------|")

    if analysis.main_package:
        layout_lines.append(
            f"| `{analysis.main_package}/` | **ACTIVE** | Main package - all source code |"
        )

    for deprecated in analysis.deprecated_paths:
        layout_lines.append(f"| `{deprecated}` | **DEPRECATED** | Legacy - DO NOT USE |")

    if (analysis.root_path / "tests").is_dir():
        layout_lines.append("| `tests/` | Active | Unit and integration tests |")

    if (analysis.root_path / "docs").is_dir():
        layout_lines.append("| `docs/` | Active | Documentation |")

    sections.append("\n".join(layout_lines) + "\n")

    # Key Components
    if analysis.key_components:
        sections.append("## Key Components\n")
        sections.append("| Component | Path | Description |")
        sections.append("|-----------|------|-------------|")

        for comp in analysis.key_components[:10]:  # Top 10
            desc = comp.docstring or f"{comp.category.title() if comp.category else 'Class'}"
            path_with_line = f"`{comp.file_path}:{comp.line_number}`"
            sections.append(f"| {comp.name} | {path_with_line} | {desc[:60]} |")

        sections.append("")

    # Common Commands
    sections.append("## Common Commands\n")
    sections.append("```bash")
    sections.append("# Install with dev dependencies")
    sections.append('pip install -e ".[dev]"')

    # Add entry points
    if analysis.entry_points:
        sections.append("")
        sections.append("# Run the application")
        for cmd in list(analysis.entry_points.keys())[:2]:
            sections.append(cmd)

    # Add dev commands
    if analysis.cli_commands:
        sections.append("")
        sections.append("# Development")
        for cmd in analysis.cli_commands:
            sections.append(cmd)

    sections.append("```\n")

    # Architecture Notes
    if analysis.architecture_patterns:
        sections.append("## Architecture\n")
        for i, pattern in enumerate(analysis.architecture_patterns, 1):
            sections.append(f"{i}. {pattern}")
        sections.append("")

    # Package Structure
    if analysis.packages:
        sections.append("## Package Structure\n")
        for pkg_name, modules in sorted(analysis.packages.items()):
            if pkg_name == "root":
                continue
            class_count = sum(len(m.classes) for m in modules)
            sections.append(f"- **{pkg_name}/**: {len(modules)} modules, {class_count} classes")
        sections.append("")

    # Important Notes
    sections.append("## Important Notes\n")
    if analysis.deprecated_paths:
        sections.append(
            f"- **ALWAYS** use `{analysis.main_package}/` not `{analysis.deprecated_paths[0]}`"
        )
    sections.append("- Check component paths above for exact file:line references")

    if analysis.config_files:
        sections.append(
            "- Key config files: " + ", ".join(f"`{f}`" for f, _ in analysis.config_files[:4])
        )

    sections.append("")

    return "\n".join(sections)


def _generate_generic_victor_md(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md for non-Python projects using language-agnostic analysis.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Generated markdown content.
    """
    context = gather_project_context(
        root_path, max_files=100, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )
    _root = Path(root_path).resolve() if root_path else Path.cwd()

    sections = []

    # Header
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    sections.append("## Project Overview\n")
    if context["readme_content"]:
        # Extract first paragraph from README
        paragraphs = context["readme_content"].split("\n\n")
        for para in paragraphs:
            stripped = para.strip()
            if stripped and not stripped.startswith(("#", "![", "<", "[!", "---", "```", "|")):
                sections.append(f"**{context['project_name']}**: {stripped[:300]}\n")
                break
        else:
            sections.append(f"**{context['project_name']}**: [Add project description here]\n")
    else:
        sections.append(f"**{context['project_name']}**: [Add project description here]\n")

    # Languages detected
    if context["detected_languages"]:
        sections.append(f"**Languages**: {', '.join(context['detected_languages'][:5])}\n")

    # Package Layout
    sections.append("## Package Layout\n")
    sections.append("| Path | Description |")
    sections.append("|------|-------------|")

    # Add directories from structure
    for dir_path in context["directory_structure"][:15]:
        if "/" not in dir_path.rstrip("/"):  # Top-level directories only
            desc = _infer_directory_purpose(dir_path.rstrip("/"))
            sections.append(f"| `{dir_path}` | {desc} |")

    sections.append("")

    # Key Files (source files)
    if context["source_files"]:
        sections.append("## Key Files\n")
        for f in context["source_files"][:15]:
            sections.append(f"- `{f}`")
        sections.append("")

    # Common Commands based on detected project type
    sections.append("## Common Commands\n")
    sections.append("```bash")

    if "pyproject.toml" in context["config_files"] or "requirements.txt" in list(
        context["config_files"]
    ):
        sections.append("# Python project")
        sections.append("pip install -r requirements.txt")
        sections.append("python main.py")
    elif "package.json" in context["config_files"]:
        sections.append("# Node.js project")
        sections.append("npm install")
        sections.append("npm start")
    elif "Cargo.toml" in context["config_files"]:
        sections.append("# Rust project")
        sections.append("cargo build")
        sections.append("cargo run")
    elif "go.mod" in context["config_files"]:
        sections.append("# Go project")
        sections.append("go build")
        sections.append("go run .")
    else:
        sections.append("# Add your build/run commands here")

    sections.append("```\n")

    # Config files found
    if context["config_files"]:
        sections.append("## Configuration\n")
        sections.append(
            "Key config files: " + ", ".join(f"`{f}`" for f in context["config_files"][:5])
        )
        sections.append("")

    # Important Notes
    sections.append("## Important Notes\n")
    sections.append("- Review and customize this file based on your project specifics")
    sections.append("- Use `/init --deep` for LLM-powered comprehensive analysis")
    sections.append("")

    return "\n".join(sections)


def _infer_directory_purpose(dirname: str) -> str:
    """Infer the purpose of a directory from its name."""
    purposes = {
        "src": "Source code",
        "lib": "Library code",
        "app": "Application code",
        "api": "API endpoints",
        "components": "UI components",
        "pages": "Page components",
        "views": "View templates",
        "models": "Data models",
        "utils": "Utility functions",
        "helpers": "Helper functions",
        "config": "Configuration",
        "configs": "Configuration",
        "tests": "Test files",
        "test": "Test files",
        "spec": "Test specifications",
        "docs": "Documentation",
        "public": "Public/static assets",
        "static": "Static files",
        "assets": "Asset files",
        "scripts": "Script files",
        "bin": "Executable scripts",
        "data": "Data files",
        "migrations": "Database migrations",
        "styles": "Stylesheets",
        "css": "CSS styles",
    }
    return purposes.get(dirname.lower(), "Project files")


# Supported context file aliases for other AI coding tools
CONTEXT_FILE_ALIASES = {
    "CLAUDE.md": "Claude Code (Anthropic)",
    "GEMINI.md": "Gemini (Google AI Studio)",
    ".cursorrules": "Cursor IDE",
    ".windsurfrules": "Windsurf IDE",
    "AGENTS.md": "Generic AI agents",
}


def create_context_symlinks(
    root_path: Optional[str] = None,
    source_file: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Create symlinks from .victor/init.md to other context file names.

    This allows a single source of truth (.victor/init.md) to work with
    multiple AI coding tools that look for different filenames.

    Args:
        root_path: Root directory. Defaults to current directory.
        source_file: Source file to link from. Defaults to settings-configured path.
        aliases: List of alias names to create. If None, creates all supported aliases.

    Returns:
        Dict mapping alias name to status ('created', 'exists', 'failed', 'skipped')
    """
    root = Path(root_path).resolve() if root_path else Path.cwd()
    # Use settings-driven path by default
    if source_file is None:
        paths = get_project_paths(root)
        source = paths.project_context_file
        source_file = str(source.relative_to(root))
    else:
        source = root / source_file
    results: Dict[str, str] = {}

    if not source.exists():
        logger.warning(f"Source file {source} does not exist")
        return {"error": f"Source file {source_file} not found"}

    # Use all aliases if not specified
    target_aliases = aliases if aliases is not None else list(CONTEXT_FILE_ALIASES.keys())

    for alias in target_aliases:
        target = root / alias
        try:
            if target.exists():
                if target.is_symlink():
                    # Check if it points to our source
                    if target.resolve() == source.resolve():
                        results[alias] = "exists"
                    else:
                        results[alias] = "exists_different"
                else:
                    results[alias] = "exists_file"
            else:
                # Create relative symlink
                target.symlink_to(source_file)
                results[alias] = "created"
                logger.info(f"Created symlink: {alias} -> {source_file}")
        except OSError as e:
            results[alias] = f"failed: {e}"
            logger.warning(f"Failed to create symlink {alias}: {e}")

    return results


def remove_context_symlinks(
    root_path: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Remove symlinks to context files.

    Only removes files that are symlinks pointing to .victor/init.md or similar.
    Does not remove actual files.

    Args:
        root_path: Root directory. Defaults to current directory.
        aliases: List of alias names to remove. If None, checks all supported aliases.

    Returns:
        Dict mapping alias name to status ('removed', 'not_symlink', 'not_found')
    """
    root = Path(root_path).resolve() if root_path else Path.cwd()
    results: Dict[str, str] = {}

    target_aliases = aliases if aliases is not None else list(CONTEXT_FILE_ALIASES.keys())

    for alias in target_aliases:
        target = root / alias
        try:
            if not target.exists() and not target.is_symlink():
                results[alias] = "not_found"
            elif target.is_symlink():
                target.unlink()
                results[alias] = "removed"
                logger.info(f"Removed symlink: {alias}")
            else:
                results[alias] = "not_symlink"
        except OSError as e:
            results[alias] = f"failed: {e}"

    return results


def _extract_readme_description(root: Path) -> str:
    """Extract project description from README."""
    readme_files = ["README.md", "README.rst", "README.txt"]

    for readme in readme_files:
        readme_path = root / readme
        if readme_path.exists():
            try:
                content = readme_path.read_text(encoding="utf-8")
                paragraphs = content.split("\n\n")

                for para in paragraphs:
                    stripped = para.strip()
                    # Skip empty, headers, images, HTML, badges
                    if not stripped:
                        continue
                    if stripped.startswith(("#", "![", "<", "[!", "---", "```", "|")):
                        continue
                    if stripped.startswith("[") and stripped.endswith(")"):
                        continue
                    # Skip lines that are only italics (like "*Any model. Any provider.*")
                    if stripped.startswith("*") and stripped.endswith("*") and "\n" not in stripped:
                        continue
                    # Strip markdown bold/italic markers to avoid formatting conflicts
                    result = stripped[:300]
                    result = result.strip("*_")  # Remove leading/trailing emphasis markers
                    result = result.replace("**", "").replace("__", "")  # Remove bold
                    return result
            except Exception:
                pass

    return ""


# =============================================================================
# LLM-Powered Analysis (Language-Agnostic)
# =============================================================================


def gather_project_context(
    root_path: Optional[str] = None,
    max_files: int = 50,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> Dict[str, any]:
    """Gather project context for LLM analysis (works with any language).

    This function collects structural information about any project type
    without parsing language-specific syntax.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        max_files: Maximum number of source files to list.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Dict containing project structure information.
    """
    root = Path(root_path).resolve() if root_path else Path.cwd()

    # Directories to skip
    skip_dirs = {
        "__pycache__",
        ".git",
        ".pytest_cache",
        "venv",
        "env",
        ".venv",
        "node_modules",
        ".tox",
        "build",
        "dist",
        "target",
        ".next",
        ".nuxt",
        "coverage",
        ".cache",
    }
    if exclude_dirs:
        skip_dirs.update(exclude_dirs)

    # Project type detection by config files
    project_indicators = {
        "pyproject.toml": "Python (modern)",
        "setup.py": "Python (legacy)",
        "package.json": "JavaScript/TypeScript",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java/Kotlin (Gradle)",
        "Gemfile": "Ruby",
        "composer.json": "PHP",
        "mix.exs": "Elixir",
        "CMakeLists.txt": "C/C++ (CMake)",
        "Makefile": "Make-based",
        "pubspec.yaml": "Dart/Flutter",
        "Package.swift": "Swift",
        ".csproj": "C# (.NET)",
    }

    # Source file extensions by language
    source_extensions = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".jsx": "JavaScript React",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".kt": "Kotlin",
        ".rb": "Ruby",
        ".php": "PHP",
        ".ex": "Elixir",
        ".exs": "Elixir",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++ Header",
        ".cs": "C#",
        ".swift": "Swift",
        ".dart": "Dart",
        ".vue": "Vue",
        ".svelte": "Svelte",
    }

    context = {
        "project_name": root.name,
        "root_path": str(root),
        "detected_languages": [],
        "config_files": [],
        "directory_structure": [],
        "source_files": [],
        "readme_content": "",
        "main_config_content": "",
    }

    # Detect project type(s)
    for config_file, lang in project_indicators.items():
        if (root / config_file).exists():
            context["detected_languages"].append(lang)
            context["config_files"].append(config_file)

    # Get README content (first 2000 chars)
    for readme in ["README.md", "README.rst", "README.txt", "readme.md"]:
        readme_path = root / readme
        if readme_path.exists():
            try:
                context["readme_content"] = readme_path.read_text(encoding="utf-8")[:2000]
            except Exception:
                pass
            break

    # Get main config content
    main_configs = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
    for config in main_configs:
        config_path = root / config
        if config_path.exists():
            try:
                context["main_config_content"] = config_path.read_text(encoding="utf-8")[:3000]
            except Exception:
                pass
            break

    # Collect directory structure (depth 2)
    def walk_dirs(path: Path, depth: int = 0, max_depth: int = 2) -> List[str]:
        dirs = []
        if depth > max_depth:
            return dirs
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith(".") or item.name in skip_dirs:
                    continue
                if item.is_dir():
                    rel_path = str(item.relative_to(root))
                    dirs.append(rel_path + "/")
                    dirs.extend(walk_dirs(item, depth + 1, max_depth))
        except PermissionError:
            pass
        return dirs

    if include_dirs:
        # Walk only included directories
        all_found_dirs = []
        for d in include_dirs:
            dir_path = root / d
            if dir_path.is_dir():
                all_found_dirs.extend(walk_dirs(dir_path))
        context["directory_structure"] = all_found_dirs[:100]
    else:
        context["directory_structure"] = walk_dirs(root)[:100]

    # Collect source files with extensions
    file_count = 0
    lang_counts: Dict[str, int] = {}

    search_paths = [root / d for d in include_dirs] if include_dirs else [root]

    for search_path in search_paths:
        if not search_path.is_dir():
            continue
        for item in search_path.rglob("*"):
            if file_count >= max_files:
                break
            if any(skip in item.parts for skip in skip_dirs):
                continue
            if item.is_file() and item.suffix in source_extensions:
                rel_path = str(item.relative_to(root))
                context["source_files"].append(rel_path)
                lang = source_extensions[item.suffix]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                file_count += 1
        if file_count >= max_files:
            break

    # Add detected languages from file extensions
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        if lang not in context["detected_languages"]:
            context["detected_languages"].append(f"{lang} ({count} files)")

    # Add content of key files
    context["key_files_content"] = {}

    # Prioritize key files
    key_file_candidates = []
    main_files = [
        f
        for f in context["source_files"]
        if "main" in f or "app" in f or "server" in f or "index" in f
    ]
    key_file_candidates.extend(main_files)

    # Add some of the largest files
    large_files = sorted(
        context["source_files"], key=lambda f: (root / f).stat().st_size, reverse=True
    )
    for f in large_files:
        if f not in key_file_candidates:
            key_file_candidates.append(f)

    key_files_to_read = key_file_candidates[:5]  # Read up to 5 key files

    for file_path in key_files_to_read:
        try:
            content = (root / file_path).read_text(encoding="utf-8")
            context["key_files_content"][file_path] = content[
                :8192
            ]  # Limit content size (fits 10-12 parallel reads)
        except Exception:
            pass

    return context


def build_llm_prompt_for_victor_md(context: Dict[str, any]) -> str:
    """Build the prompt for LLM to generate project context file.

    Args:
        context: Project context from gather_project_context()

    Returns:
        Prompt string for the LLM.
    """

    # Static part of the prompt (the "80%")
    prompt_header = f"""You are an expert software architect tasked with creating a high-level "user manual" for an AI coding assistant named Victor.
This manual, named {VICTOR_CONTEXT_FILE}, will help Victor understand the project's structure, purpose, and conventions.
Your analysis must be comprehensive, distilling the provided information into a clear and actionable guide.

Analyze the following project data and generate the {VICTOR_CONTEXT_FILE} file.

**Output Rules:**
1.  **Start with the Header**: The response MUST begin with `# {VICTOR_CONTEXT_FILE}`.
2.  **Use Markdown**: Format the entire output in clean, readable Markdown. Use tables for structured data.
3.  **Be Factual**: Base your analysis exclusively on the provided context. Do not infer or add information not present in the data.
4.  **Be Concise**: Provide high-level summaries. Focus on the "what" and "why," not implementation details.
5.  **Follow the Structure**: Generate all of the requested sections.

---
"""

    # Dynamic part of the prompt (the "20%")
    dynamic_context = f"""
**Project Name**: {context['project_name']}
**Detected Languages**: {', '.join(context['detected_languages']) or 'Unknown'}

**Configuration Files**:
{chr(10).join('- ' + f for f in context['config_files']) or 'None detected'}

**Directory Structure Overview**:
```
{chr(10).join(context['directory_structure'][:50]) or 'Unable to determine'}
```

**Sample of Source Files**:
```
{chr(10).join(context['source_files'][:30]) or 'No source files found'}
```
"""

    if context.get("key_files_content"):
        dynamic_context += "\n**Content of Key Files**:\n"
        for file_path, content in context["key_files_content"].items():
            dynamic_context += f"--- `{file_path}` ---\n```\n{content}\n```\n\n"

    # Static part of the prompt (continued)
    prompt_footer = f"""
---

**Generation Task**:

Generate the full content for the `{VICTOR_CONTEXT_FILE}` file, adhering to all rules above. Create the following sections:

1.  `## Project Overview`
    -   Write a one-paragraph summary of the project's purpose, based on the README and file structure.

2.  `## Package Layout`
    -   Create a Markdown table with columns: `| Path | Status | Description |`.
    -   List the most important top-level directories.
    -   Infer the purpose of each directory (e.g., source code, tests, docs). Mark the main source directory as `**ACTIVE**`.

3.  `## Key Components`
    -   Identify 5-7 key files or classes from the provided context.
    -   Create a Markdown table with columns: `| Component | Path | Description |`.
    -   Provide a one-sentence description for each component.

4.  `## Common Commands`
    -   Based on the configuration files (e.g., `package.json`, `pyproject.toml`), list the essential commands for building, testing, and running the project inside a `bash` code block.

5.  `## Architecture Notes`
    -   From the file names and structure, infer 2-3 high-level architectural patterns. (e.g., "Provider Pattern for multiple LLMs", "REST API with FastAPI", "CLI application using Typer").

6.  `## Important Notes`
    -   Add 2-3 bullet points for an AI assistant to remember, such as "Always use `victor/` for core source code" or "Check `pyproject.toml` for dependencies".

Remember, output ONLY the generated `{VICTOR_CONTEXT_FILE}` content, starting with the `# {VICTOR_CONTEXT_FILE}` header.
"""

    return prompt_header + dynamic_context + prompt_footer


async def generate_victor_md_with_llm(
    provider,
    model: str,
    root_path: Optional[str] = None,
    max_files: int = 50,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate project context file using an LLM provider.

    This function works with any project type by gathering structural
    information and asking the LLM to analyze and document it.

    Args:
        provider: A Victor provider instance (BaseProvider)
        model: Model identifier to use for generation
        root_path: Root directory to analyze. Defaults to current directory.
        max_files: Maximum source files to include in context.
        include_dirs: List of directories to include in the analysis.
        exclude_dirs: List of directories to exclude from the analysis.

    Returns:
        Generated content for .victor/init.md.
    """
    from victor_contracts.provider_runtime import Message

    # Gather project context
    context = gather_project_context(
        root_path, max_files, include_dirs=include_dirs, exclude_dirs=exclude_dirs
    )

    # Build prompt
    prompt = build_llm_prompt_for_victor_md(context)

    # Call the LLM
    messages = [Message(role="user", content=prompt)]

    expected_header = f"# {VICTOR_CONTEXT_FILE}"

    try:
        response = await provider.chat(messages, model=model)
        content = response.content.strip()

        # Ensure it starts with the header
        if not content.startswith(expected_header):
            content = f"{expected_header}\\n\\n" + content

        return content
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fall back to basic generation
        return generate_smart_victor_md(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )


def _collect_embedding_status(
    root_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Summarize embedding/cache health for init.md enrichment."""
    root = Path(root_path).resolve() if root_path else Path.cwd()
    try:
        from victor.storage.cache.embedding_cache_manager import EmbeddingCacheManager
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        logger.debug("Embedding manager unavailable: %s", exc)
        return None

    try:
        status = EmbeddingCacheManager.get_instance().get_status()
    except Exception as exc:  # pragma: no cover - runtime I/O
        logger.debug("Failed to collect embedding status: %s", exc)
        return None

    caches: List[Dict[str, Any]] = []
    for cache in status.caches:
        caches.append(
            {
                "name": cache.name,
                "type": cache.cache_type.value,
                "files": cache.file_count,
                "size": cache.size_str,
                "age": cache.age_str,
            }
        )

    info: Dict[str, Any] = {
        "total_files": status.total_files,
        "total_size": status.total_size_str,
        "caches": caches,
    }

    # Optional: inspect code embedding store (LanceDB) for metadata richness
    try:
        from victor_coding.compat.settings import get_project_paths, load_settings

        settings = load_settings()
        default_dir = get_project_paths(root).embeddings_dir
        persist_dir = Path(getattr(settings, "codebase_persist_directory", None) or default_dir)

        import lancedb  # type: ignore

        if persist_dir.exists():
            db = lancedb.connect(str(persist_dir))
            table_names = db.list_tables().tables
            table_name = getattr(settings, "codebase_embedding_table", None) or (
                table_names[0] if table_names else None
            )
            if table_name:
                table = db.open_table(table_name)
                row_count = table.count_rows()

                sample_keys: Dict[str, int] = defaultdict(int)
                chunk_types: Set[str] = set()
                span_lengths: List[int] = []

                try:
                    sample_rows = table.head(min(50, row_count)).to_list()  # small sample
                except Exception:
                    sample_rows = []

                for row in sample_rows:
                    for key, value in row.items():
                        if key in {"id", "vector"}:
                            continue
                        if value not in (None, "", [], {}):
                            sample_keys[key] += 1
                        if key == "chunk_type" and isinstance(value, str):
                            chunk_types.add(value)
                        if key in {"line_start", "line_end"}:
                            try:
                                start = int(row.get("line_start", 0))
                                end = int(row.get("line_end", 0))
                                if start and end and end >= start:
                                    span_lengths.append(end - start + 1)
                            except Exception:
                                pass

                info["code_embeddings"] = {
                    "path": str(persist_dir),
                    "table": table_name,
                    "rows": row_count,
                    "metadata_keys": sorted(sample_keys.items(), key=lambda x: -x[1])[:8],
                    "chunk_types": sorted(chunk_types)[:6] if chunk_types else [],
                    "max_span": max(span_lengths) if span_lengths else None,
                }
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("Skipping code embedding metadata inspection: %s", exc)

    return info


def _build_analyzer_section(
    stats: Dict[str, Any],
    graph_insights: Dict[str, Any],
    embedding_status: Optional[Dict[str, Any]],
) -> List[str]:
    """Construct analyzer coverage section from index + graph data."""
    if not stats:
        return []

    lines: List[str] = []
    lines.append("## Analyzer Coverage\n")

    total_symbols = stats.get("total_symbols", 0)
    total_files = stats.get("total_files", 0)
    lines.append(
        f"- **Symbol index**: {total_symbols} symbols across {total_files} files "
        "(multi-language tree-sitter + regex fallback, includes call sites and references)"
    )

    if graph_insights.get("has_graph"):
        graph_stats = graph_insights.get("stats", {})
        edge_types = graph_stats.get("edge_types", {})
        call_edges = edge_types.get("CALLS", 0)
        ref_edges = edge_types.get("REFERENCES", 0)
        import_edges = edge_types.get("IMPORTS", 0)
        inherits = edge_types.get("INHERITS", 0)
        comp = edge_types.get("COMPOSED_OF", 0)

        lines.append(
            f"- **Code graph**: {graph_stats.get('total_nodes', 0)} nodes, "
            f"{graph_stats.get('total_edges', 0)} edges "
            f"(calls: {call_edges}, refs: {ref_edges}, imports: {import_edges}, "
            f"inheritance: {inherits}, composition: {comp}); module PageRank and coupling detection ready"
        )

        # Highlight a few hubs and module ranks for quick navigation
        hubs = graph_insights.get("hub_classes", [])[:2]
        if hubs:
            hub_preview = ", ".join(f"`{hub['name']}` ({hub['degree']} links)" for hub in hubs)
            lines.append(f"  - Hub classes: {hub_preview}")

        modules = graph_insights.get("important_modules", [])[:3]
        if modules:
            module_preview = ", ".join(f"`{mod['module']}` ({mod['role']})" for mod in modules)
            lines.append(f"  - Module PageRank leaders: {module_preview}")

        languages = graph_insights.get("languages", [])
        if languages:
            top_langs = ", ".join(f"{lang} ({count})" for lang, count in languages[:3])
            lines.append(f"  - Graph coverage by language: {top_langs}")

        if edge_types:
            missing_edges = graph_insights.get("edge_gaps", [])
            if missing_edges:
                lines.append(
                    f"  - Missing edge types: {', '.join(missing_edges)} (re-run index with tree-sitter deps to capture)"
                )
            elif call_edges == 0:
                lines.append(
                    "  - Calls not captured; verify tree-sitter call extraction is installed."
                )

        call_hotspots = graph_insights.get("call_hotspots", [])
        if call_hotspots:
            hot_preview = ", ".join(
                f"`{sym['name']}` ({sym['in_degree']} callers)" for sym in call_hotspots[:3]
            )
            lines.append(f"  - Call hotspots: {hot_preview}")
    else:
        lines.append(
            "- **Code graph**: not detected yet (run `victor index` to populate tree-sitter call/reference graph)"
        )

    if embedding_status and embedding_status.get("caches"):
        cache_preview = ", ".join(
            f"{cache['name']} ({cache['files']} files, {cache['size']}, {cache['age']})"
            for cache in embedding_status["caches"][:3]
        )
        lines.append(
            f"- **Semantic embeddings**: {cache_preview} "
            "(tree-sitter chunking for code spans; tool/intent/conversation caches ready)"
        )
        if embedding_status.get("code_embeddings"):
            ce = embedding_status["code_embeddings"]
            meta_keys = ", ".join(k for k, _ in ce.get("metadata_keys", [])) or "none"
            lines.append(
                f"  - Code embeddings: {ce.get('rows', 0)} vectors @ {ce.get('path')} "
                f"(metadata keys: {meta_keys})"
            )

    lines.append("")
    return lines


def _infer_python_requires(root: Path) -> Optional[str]:
    """Read requires-python from pyproject.toml if present."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return None
    try:
        content = pyproject.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"requires-python\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]", content)
        if match:
            return match.group(1)
    except Exception:
        return None
    return None


def _infer_commands(root: Path) -> List[str]:
    """Infer project commands from Makefile, package.json, and common conventions."""
    commands: List[str] = []

    makefile = root / "Makefile"
    if makefile.exists():
        try:
            make_text = makefile.read_text(encoding="utf-8", errors="ignore")
            targets = set(re.findall(r"^(\\w+):", make_text, re.MULTILINE))
            preferred = [
                ("install-dev", "make install-dev"),
                ("install", "make install"),
                ("lint", "make lint"),
                ("format", "make format"),
                ("test", "make test"),
                ("test-all", "make test-all"),
                ("serve", "make serve"),
                ("build", "make build"),
                ("docker", "make docker"),
            ]
            for target, cmd in preferred:
                if target in targets:
                    commands.append(cmd)
        except Exception:
            pass

    pkg = root / "package.json"
    if pkg.exists():
        try:
            import json

            data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
            scripts = data.get("scripts", {}) or {}
            for key in ("dev", "start", "build", "test", "lint"):
                if key in scripts:
                    commands.append(f"npm run {key}")
            commands.append("npm install")
        except Exception:
            pass

    # Python defaults
    commands.append('pip install -e ".[dev]"')
    commands.append("pytest")

    # Docker hints
    if (root / "docker-compose.yml").exists():
        commands.append("docker-compose up -d")

    # Server/UI hints
    if (root / "web" / "server").exists():
        commands.append("uvicorn web.server.main:app --reload")
    if (root / "web" / "ui").exists():
        commands.append("cd web/ui && npm install && npm run dev")

    # Deduplicate preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for cmd in commands:
        if cmd not in seen:
            deduped.append(cmd)
            seen.add(cmd)
    return deduped


def _infer_env_vars(root: Path) -> List[str]:
    """Infer env vars from .env/.env.example (uppercase keys)."""
    env_files = [root / ".env", root / ".env.example"]
    vars_found: List[str] = []
    seen: Set[str] = set()
    for env_file in env_files:
        if not env_file.exists():
            continue
        try:
            for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key = line.split("=", 1)[0].strip()
                if key.isupper() and key not in seen:
                    vars_found.append(key)
                    seen.add(key)
        except Exception:
            continue
    return vars_found[:10]


def _build_quick_start(commands: List[str]) -> List[str]:
    """Pick a concise quick-start command set."""
    if not commands:
        return []
    quick: List[str] = []
    # Prefer install, lint/test, serve/dev commands if present
    priorities = [
        ("install", ("install", "pip install", "npm install", "make install")),
        ("lint", ("lint", "ruff", "eslint")),
        ("test", ("test", "pytest", "npm test", "make test")),
        ("serve", ("serve", "uvicorn", "run dev", "npm run dev")),
    ]
    lower_cmds = [(c.lower(), c) for c in commands]
    for _label, needles in priorities:
        for lc, orig in lower_cmds:
            if any(n in lc for n in needles):
                if orig not in quick:
                    quick.append(orig)
                break
    # Fallback: first 3 commands
    for cmd in commands:
        if len(quick) >= 3:
            break
        if cmd not in quick:
            quick.append(cmd)
    return quick[:4]


def _find_config_files(root: Path) -> List[str]:
    """Find common config files (json/yaml/toml) while skipping vendor/venv/build artifacts."""
    exts = {".json", ".yaml", ".yml", ".toml"}
    results: List[str] = []
    extra_skip = {
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        "out",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
    skip_parts = {"htmlcov", "htmlcov_lang", "coverage", "__pycache__", "egg-info"}

    for path in root.rglob("*"):
        if path.suffix not in exts or not path.is_file():
            continue
        if should_ignore_path(path, skip_dirs=DEFAULT_SKIP_DIRS, extra_skip_dirs=extra_skip):
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        rel = str(path.relative_to(root))
        results.append(rel)
        if len(results) >= 40:  # cap for speed
            break

    results.sort()
    return results[:15]


def _find_docs_files(root: Path) -> List[str]:
    """Find markdown/adoc docs (top-N) while skipping vendor/venv/build artifacts."""
    results: List[tuple[int, str]] = []
    extra_skip = {
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        "out",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
    skip_parts = {"htmlcov", "htmlcov_lang", "coverage", "__pycache__", "egg-info"}

    for path in root.rglob("*.md"):
        if should_ignore_path(path, skip_dirs=DEFAULT_SKIP_DIRS, extra_skip_dirs=extra_skip):
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        rel = str(path.relative_to(root))
        results.append((size, rel))
        if len(results) >= 80:  # reasonable cap
            break
    results.sort(reverse=True)  # bigger docs first
    return [rel for _size, rel in results[:12]]


async def generate_victor_md_from_index(
    root_path: Optional[str] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md from the SymbolStore (pre-indexed symbols).

    This uses the SQLite symbol store for fast, accurate init.md generation.
    The symbol store should be indexed first via `/init --index` or automatically
    during first run.

    Args:
        root_path: Root directory. Defaults to current directory.
        force: If True, re-index all files ignoring cache.

    Returns:
        Generated markdown content for .victor/init.md.
    """
    from victor_coding.codebase.symbol_store import SymbolStore

    root = Path(root_path).resolve() if root_path else Path.cwd()
    store = SymbolStore(str(root), include_dirs=include_dirs, exclude_dirs=exclude_dirs)

    # Index if needed (quick operation if already indexed, unless force=True)
    await store.index_codebase(force=force)

    stats = store.get_stats()
    key_components = store.find_key_components(limit=20)  # Increased from 15 for more context
    store.get_detected_patterns()
    store.find_named_implementations()
    store.find_performance_hints()

    # Also run CodebaseAnalyzer for enhanced info (dependencies, LOC, imports, coverage)
    analyzer = CodebaseAnalyzer(str(root), include_dirs=include_dirs, exclude_dirs=exclude_dirs)
    analyzer._extract_dependencies()
    analyzer._calculate_loc_stats()
    analyzer._extract_top_imports()
    analyzer._extract_test_coverage()
    enhanced_info = analyzer.analysis

    graph_insights = await extract_graph_insights(root_path)
    _collect_embedding_status(root_path)
    env_vars = _infer_env_vars(root)
    commands_inferred = _infer_commands(root)
    quick_start = _build_quick_start(commands_inferred)
    config_files = _find_config_files(root)
    docs_files = _find_docs_files(root)

    return _build_init_md_content(
        root=root,
        stats=stats,
        key_components=key_components,
        enhanced_info=enhanced_info,
        graph_insights=graph_insights,
        env_vars=env_vars,
        quick_start=quick_start,
        config_files=config_files,
        docs_files=docs_files,
    )


def _build_init_md_content(
    root: Path,
    stats: Dict[str, Any],
    key_components: List[Any],
    enhanced_info: Any,
    graph_insights: Dict[str, Any],
    env_vars: List[str],
    quick_start: List[str],
    config_files: List[str],
    docs_files: List[str],
) -> str:
    """Render all collected data into init.md markdown.

    Shared by generate_victor_md_from_index (SymbolStore path) and
    generate_victor_md_from_graph (graph RAG path) so content is identical
    regardless of which indexing backend provided the symbol data.
    """
    sections = []

    # Header
    sections.append(f"# {VICTOR_CONTEXT_FILE}\n")
    sections.append(
        "This file provides guidance to Victor when working with code in this repository.\n"
    )

    # Project Overview
    sections.append("## Project Overview\n")
    readme_desc = _extract_readme_description(root)
    if readme_desc:
        sections.append(f"**{root.name}**: {readme_desc}\n")
    else:
        sections.append(f"**{root.name}**: [Add project description here]\n")

    # Languages
    if stats.get("files_by_language"):
        langs = [f"{lang} ({count})" for lang, count in stats["files_by_language"].items()]
        sections.append(f"**Languages**: {', '.join(langs)}\n")

    # Package Layout
    sections.append("## Package Layout\n")
    sections.append("| Path | Type | Description |")
    sections.append("|------|------|-------------|")

    dirs_seen: Set[str] = set()

    def add_dir(path: str, type_label: str, desc: str) -> None:
        if path in dirs_seen:
            return
        if (root / path).exists():
            sections.append(f"| `{path}/` | {type_label} | {desc} |")
            dirs_seen.add(path)

    add_dir("victor", "**ACTIVE**", "Backend / core")
    add_dir("src", "Active", "Source code")
    add_dir("web/server", "Active", "Backend server")
    add_dir("web/ui", "Active", "Web UI")
    add_dir("vscode-victor", "Active", "Editor extension")
    add_dir("docs", "Active", "Documentation")
    add_dir("docs/guides", "Active", "Guides and playbooks")
    add_dir("examples", "Active", "Examples and sample workflows")
    add_dir("scripts", "Active", "Automation and helper scripts")
    add_dir("templates", "Active", "Scaffold and template files")
    add_dir("tests", "Active", "Unit and integration tests")
    add_dir("victor_test", "Active", "Lightweight demos")
    add_dir("archive", "Legacy", "Historical code (frozen)")

    for comp in key_components:
        dir_parts = Path(comp.file_path).parts
        if len(dir_parts) > 1:
            add_dir(dir_parts[0], "**ACTIVE**", "Source code")

    sections.append("")

    # Key Components
    if key_components:
        sections.append("## Key Components\n")
        sections.append("| Component | Type | Path | Description |")
        sections.append("|-----------|------|------|-------------|")

        filtered_components = [
            comp
            for comp in key_components
            if not str(comp.file_path).startswith("tests/")
            and not str(comp.file_path).startswith("vscode-victor/out")
        ]
        display_components = filtered_components or key_components
        seen_components: Set[str] = set()

        for comp in display_components[:15]:
            if comp.name in seen_components:
                continue
            seen_components.add(comp.name)
            raw_desc = (
                comp.docstring or comp.category.title()
                if comp.category
                else comp.symbol_type.title()
            )
            desc = raw_desc.split("\n")[0][:120].strip()
            if len(raw_desc) > 120 and "." in desc:
                desc = desc.rsplit(".", 1)[0] + "."
            path_with_line = f"`{comp.file_path}:{comp.line_number}`"
            sections.append(f"| {comp.name} | {comp.symbol_type} | {path_with_line} | {desc} |")

        sections.append("")

    # Dependencies (from pyproject.toml/package.json)
    if enhanced_info.dependencies:
        sections.append("## Dependencies\n")
        if enhanced_info.dependencies.get("core"):
            core_deps = enhanced_info.dependencies["core"][:12]
            sections.append(
                f"**Core** ({len(enhanced_info.dependencies['core'])} packages): {', '.join(core_deps)}"
            )
            if len(enhanced_info.dependencies["core"]) > 12:
                sections.append(f"  ...and {len(enhanced_info.dependencies['core']) - 12} more")
        if enhanced_info.dependencies.get("dev"):
            dev_deps = enhanced_info.dependencies["dev"][:8]
            sections.append(
                f"\n**Dev** ({len(enhanced_info.dependencies['dev'])} packages): {', '.join(dev_deps)}"
            )
        sections.append("")

    # Configuration hints
    sections.append("## Configuration\n")
    sections.append(
        "- Settings: `.env` → `~/.victor/profiles.yaml` → CLI flags (override order)\n"
        "- Project context: `.victor/init.md` (regenerate with `victor init --update`)"
    )
    sections.append("")

    if quick_start:
        sections.append("## Quick Start\n")
        sections.append("```bash")
        for cmd in quick_start:
            sections.append(cmd)
        sections.append("```\n")

    if env_vars:
        sections.append("## Environment Variables\n")
        sections.append("Likely used (from .env/.env.example):")
        for var in env_vars:
            sections.append(f"- `{var}`")
        sections.append("")

    if config_files:
        sections.append("## Config Files\n")
        for cfg in config_files[:12]:
            sections.append(f"- `{cfg}`")
        sections.append("")

    if docs_files:
        sections.append("## Documentation\n")
        for doc in docs_files[:10]:
            sections.append(f"- `{doc}`")
        sections.append("")

    # Codebase Stats (LOC, files, coverage)
    if enhanced_info.loc_stats:
        sections.append("## Codebase Stats\n")
        loc = enhanced_info.loc_stats
        total_lines = loc.get("total_lines", 0)
        total_file_count = loc.get("total_files", 0)
        stats_line = f"- **{total_lines:,}** lines of code across **{total_file_count}** files"
        if enhanced_info.test_coverage is not None:
            stats_line += f" ({enhanced_info.test_coverage}% test coverage)"
        sections.append(stats_line)

        if loc.get("source_files") and loc.get("config_files"):
            sections.append(
                f"  - **Source**: {loc.get('source_lines', 0):,} LOC in {loc.get('source_files', 0)} files"
            )
            sections.append(
                f"  - **Config**: {loc.get('config_lines', 0):,} LOC in {loc.get('config_files', 0)} files"
            )
        sections.append("")

    # Hub classes (high connectivity)
    if graph_insights.get("has_graph") and graph_insights.get("hub_classes"):
        sections.append("## Hub Classes\n")
        for hub in graph_insights["hub_classes"][:5]:
            line_ref = f":{hub['line']}" if hub.get("line") else ""
            sections.append(
                f"- `{hub['name']}` (`{hub['file']}{line_ref}`) — {hub['degree']} connections"
            )
        sections.append("")

    # Key architectural modules
    if graph_insights.get("has_graph") and graph_insights.get("important_modules"):
        sections.append("## Key Modules\n")
        for mod in graph_insights["important_modules"][:5]:
            conns = f"↓{mod['in_degree']} ↑{mod['out_degree']}"
            sections.append(f"- `{mod['module']}` ({mod['role']}, {conns})")
        sections.append("")

    sections.append("## Notes\n")
    sections.append("- Run `/init --update` to refresh after code changes")
    sections.append("")

    return "\n".join(sections)


async def generate_victor_md_from_graph(
    root_path: Optional[str] = None,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> str:
    """Generate init.md from the graph RAG index (project.db graph_node table).

    Fast path used when ``victor init --ccg`` has already built the graph.
    Skips SymbolStore.index_codebase() entirely — reads pre-built symbol nodes
    from ``graph_node`` via SQL queries and feeds them into the same content
    builder as ``generate_victor_md_from_index``.

    The graph RAG index is a strict superset of the SymbolStore:
    - All symbol types (class, function, method, interface, struct)
    - Richer metadata (signature, docstring, visibility, ast_kind, lang)
    - Cross-file edges (CALLS, CONTAINS, INHERITS, IMPLEMENTS, CFG/CDG/DDG)
    """
    import sqlite3
    from victor_coding.codebase.symbol_store import SymbolInfo, ARCHITECTURE_PATTERNS

    root = Path(root_path).resolve() if root_path else Path.cwd()
    graph_db_path = root / ".victor" / "project.db"

    stats: Dict[str, Any] = {
        "files_by_language": {},
        "total_symbols": 0,
        "total_files": 0,
    }
    key_components: List[Any] = []

    if graph_db_path.exists():
        conn = sqlite3.connect(str(graph_db_path))
        try:
            # Language distribution (COUNT GROUP BY — no per-row Python objects)
            lang_rows = conn.execute(
                "SELECT lang, COUNT(*) FROM graph_node WHERE lang IS NOT NULL GROUP BY lang ORDER BY 2 DESC"
            ).fetchall()
            stats["files_by_language"] = {r[0]: r[1] for r in lang_rows if r[0]}

            total_row = conn.execute("SELECT COUNT(*) FROM graph_node").fetchone()
            stats["total_symbols"] = total_row[0] if total_row else 0

            file_row = conn.execute(
                "SELECT COUNT(DISTINCT file) FROM graph_node WHERE file IS NOT NULL"
            ).fetchone()
            stats["total_files"] = file_row[0] if file_row else 0

            # Key components: prioritise classes/interfaces, then sort by connectivity
            rows = conn.execute("""
                SELECT n.name, n.type, n.file, n.line, n.lang, n.signature, n.docstring,
                       (SELECT COUNT(*) FROM graph_edge WHERE src = n.node_id OR dst = n.node_id) AS degree
                FROM graph_node n
                WHERE n.type IN ('class', 'interface', 'struct', 'function', 'method')
                  AND n.name NOT LIKE '<%'
                  AND n.file IS NOT NULL
                ORDER BY
                  CASE n.type WHEN 'class' THEN 1 WHEN 'interface' THEN 2
                               WHEN 'struct' THEN 3 ELSE 4 END,
                  degree DESC
                LIMIT 25
            """).fetchall()

            for (
                name,
                sym_type,
                file_path,
                line,
                lang,
                signature,
                docstring,
                _degree,
            ) in rows:
                if not name or not file_path:
                    continue
                try:
                    rel_path = str(Path(file_path).relative_to(root))
                except ValueError:
                    rel_path = file_path

                # Infer architecture category from name suffix (same heuristic as SymbolStore)
                category = None
                for cat, patterns in ARCHITECTURE_PATTERNS.items():
                    if any(p.lower() in name.lower() for p in patterns):
                        category = cat
                        break

                key_components.append(
                    SymbolInfo(
                        name=name,
                        symbol_type=sym_type,
                        file_path=rel_path,
                        line_number=line or 0,
                        language=lang or "",
                        category=category,
                        docstring=docstring,
                        signature=signature,
                    )
                )
        finally:
            conn.close()

    # Shared context: LOC, deps, env vars, etc. — fast file-level scans, no symbol re-parsing
    analyzer = CodebaseAnalyzer(str(root), include_dirs=include_dirs, exclude_dirs=exclude_dirs)
    analyzer._extract_dependencies()
    analyzer._calculate_loc_stats()
    analyzer._extract_top_imports()
    analyzer._extract_test_coverage()
    enhanced_info = analyzer.analysis

    graph_insights = await extract_graph_insights(root_path)
    env_vars = _infer_env_vars(root)
    commands_inferred = _infer_commands(root)
    quick_start = _build_quick_start(commands_inferred)
    config_files = _find_config_files(root)
    docs_files = _find_docs_files(root)

    return _build_init_md_content(
        root=root,
        stats=stats,
        key_components=key_components,
        enhanced_info=enhanced_info,
        graph_insights=graph_insights,
        env_vars=env_vars,
        quick_start=quick_start,
        config_files=config_files,
        docs_files=docs_files,
    )


async def extract_conversation_insights(
    root_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract insights from conversation history to enhance init.md.

    Analyzes stored conversations to identify:
    - Frequently asked questions/topics
    - Common file references
    - Learned patterns and hot spots

    Args:
        root_path: Root directory containing .victor/project.db with sessions/messages tables

    Returns:
        Dictionary with extracted insights
    """
    import sqlite3
    from collections import Counter

    root = Path(root_path).resolve() if root_path else Path.cwd()
    db_path = (
        root / ".victor" / "project.db"
    )  # Fixed: conversations are in project.db, not conversation.db

    if not db_path.exists():
        return {"error": "No conversation history found"}

    insights = {
        "common_topics": [],
        "hot_files": [],
        "faq": [],
        "learned_patterns": [],
        "session_count": 0,
        "message_count": 0,
    }

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get session and message counts
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
        insights["session_count"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        insights["message_count"] = cursor.fetchone()[0]

        # Extract common user queries (deduplicated, excluding benchmarks)
        cursor.execute("""
            SELECT content FROM messages
            WHERE role = 'user'
              AND content NOT LIKE '%<TOOL_OUTPUT%'
              AND content NOT LIKE '%Complete this Python function%'
              AND content NOT LIKE '%Complete the following Python%'
              AND length(content) BETWEEN 20 AND 500
        """)

        queries = [row[0] for row in cursor.fetchall()]
        query_counter = Counter()

        # Extract key topics from queries
        topic_keywords = [
            "component",
            "architecture",
            "test",
            "bug",
            "fix",
            "add",
            "create",
            "refactor",
            "improve",
            "explain",
            "how",
            "what",
            "why",
            "where",
            "error",
            "issue",
            "implement",
            "feature",
            "config",
            "setup",
        ]

        for query in queries:
            query_lower = query.lower()
            for keyword in topic_keywords:
                if keyword in query_lower:
                    query_counter[keyword] += 1

        insights["common_topics"] = query_counter.most_common(10)

        # Extract frequently referenced files from assistant responses
        cursor.execute("""
            SELECT content FROM messages
            WHERE role = 'assistant'
              AND (content LIKE '%.py%' OR content LIKE '%.ts%' OR content LIKE '%.js%')
        """)

        file_counter = Counter()
        file_pattern = re.compile(r"`([a-zA-Z_/]+\.(py|ts|js|go|rs))[:`]")

        # Directories that indicate SWE-bench/benchmark test data, not project files
        _BENCHMARK_PREFIXES = (
            "astropy/",
            "django/",
            "flask/",
            "requests/",
            "sphinx/",
            "sympy/",
            "matplotlib/",
            "scikit",
            "pandas/",
            "numpy/",
            "pylint/",
            "pytest/",
            "httpx/",
            "pydantic/",
        )

        for row in cursor.fetchall():
            matches = file_pattern.findall(row[0])
            for match in matches:
                file_path = match[0]
                # Normalize and count — skip benchmark/external repo files
                if "/" in file_path and not file_path.startswith("/"):
                    if not file_path.startswith(_BENCHMARK_PREFIXES):
                        file_counter[file_path] += 1

        insights["hot_files"] = file_counter.most_common(15)

        # Get architectural patterns from patterns table
        cursor.execute("""
            SELECT pattern_name, pattern_type, COUNT(*) as count
            FROM patterns
            GROUP BY pattern_type
            ORDER BY count DESC
        """)
        insights["learned_patterns"] = [
            {"name": row[0], "type": row[1], "count": row[2]} for row in cursor.fetchall()
        ]

        # Extract FAQ-like questions (questions asked multiple times)
        # Filter out benchmark/test prompts and trivial questions
        cursor.execute("""
            SELECT content, COUNT(*) as times
            FROM messages
            WHERE role = 'user'
              AND content LIKE '%?%'
              AND content NOT LIKE '%Complete%function%'
              AND content NOT LIKE '%What is 2 + 2%'
              AND content NOT LIKE '%What is Python%'
              AND content NOT LIKE '%Reply with just%'
              AND content NOT LIKE '%<TOOL_OUTPUT%'
              AND length(content) BETWEEN 30 AND 200
            GROUP BY content
            HAVING times > 1
            ORDER BY times DESC
            LIMIT 5
        """)
        insights["faq"] = [{"question": row[0], "times_asked": row[1]} for row in cursor.fetchall()]

        conn.close()

    except Exception as e:
        insights["error"] = str(e)

    return insights


def _get_init_usage_logger():
    """Get or create a UsageLogger for init command operations.

    Writes to the same usage.jsonl as chat sessions so GEPA can
    learn from init LLM synthesis quality over time.
    """
    try:
        from victor.observability.analytics.logger import UsageLogger
        from victor_coding.compat.settings import get_project_paths

        paths = get_project_paths()
        logs_dir = paths.logs_dir if hasattr(paths, "logs_dir") else paths.victor_dir / "logs"
        from pathlib import Path

        logs_dir = Path(str(logs_dir))
        logs_dir.mkdir(parents=True, exist_ok=True)
        return UsageLogger(log_file=logs_dir / "usage.jsonl", enabled=True)
    except Exception:
        # Fallback: no-op logger
        class _NoOpLogger:
            session_id = "init"

            def log_event(self, *a, **kw):
                pass

        return _NoOpLogger()


async def extract_graph_insights(root_path: Optional[str] = None) -> Dict[str, Any]:
    """Extract insights from the code graph for init.md enrichment.

    Analyzes the code graph to detect:
    - Design patterns (Provider, Factory, Facade, etc.)
    - Most important symbols (PageRank)
    - Hub classes (high centrality)
    - File dependencies
    - Graph statistics

    Args:
        root_path: Root directory containing .victor/graph

    Returns:
        Dictionary with graph insights
    """
    from pathlib import Path

    try:
        from victor.tools.graph_tool import GraphAnalyzer, _load_graph
    except ImportError:
        GraphAnalyzer = None
        _load_graph = None
    from victor_coding.codebase.graph.registry import create_graph_store

    root = Path(root_path).resolve() if root_path else Path.cwd()
    # Use consolidated project.db
    graph_db_path = root / ".victor" / "project.db"
    _NT = "graph_node"
    _ET = "graph_edge"

    insights: Dict[str, Any] = {
        "has_graph": False,
        "patterns": [],
        "important_symbols": [],
        "hub_classes": [],
        "stats": {},
        # Module-level insights
        "important_modules": [],
        "module_coupling": [],
        "languages": [],
        "call_hotspots": [],
        "edge_gaps": [],
        "pagerank": [],
        "centrality": [],
        "components": [],
    }

    if not graph_db_path.exists():
        return insights

    try:
        # Use direct SQL queries for fast stats instead of loading entire graph
        import sqlite3

        conn = sqlite3.connect(graph_db_path)
        try:
            # Get basic stats
            cur = conn.execute(f"SELECT COUNT(*) FROM {_NT}")  # nosemgrep
            total_nodes = cur.fetchone()[0]

            if total_nodes == 0:
                return insights

            cur = conn.execute(f"SELECT COUNT(*) FROM {_ET}")  # nosemgrep
            total_edges = cur.fetchone()[0]

            # Get node type distribution
            cur = conn.execute(f"SELECT type, COUNT(*) FROM {_NT} GROUP BY type")  # nosemgrep
            node_types = dict(cur.fetchall())

            # Language coverage
            cur = conn.execute(f"SELECT lang, COUNT(*) FROM {_NT} GROUP BY lang")  # nosemgrep
            languages = [(row[0], row[1]) for row in cur.fetchall() if row[0]]

            # Get edge type distribution
            cur = conn.execute(f"SELECT type, COUNT(*) FROM {_ET} GROUP BY type")  # nosemgrep
            edge_types = dict(cur.fetchall())

            insights["has_graph"] = True
            insights["stats"] = {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_types": node_types,
                "edge_types": edge_types,
            }
            insights["languages"] = languages
            # Identify missing edge types for debugging coverage
            expected_edges = {
                "CALLS",
                "REFERENCES",
                "IMPORTS",
                "INHERITS",
                "IMPLEMENTS",
                "COMPOSED_OF",
            }
            insights["edge_gaps"] = sorted(expected_edges - set(edge_types.keys()))

            # Get high-connectivity nodes (hub classes) via SQL
            # fmt: off
            cur = conn.execute(  # nosemgrep
                f"""
                SELECT n.name, n.type, n.file, n.line,
                       (SELECT COUNT(*) FROM {_ET} WHERE src = n.node_id) +
                       (SELECT COUNT(*) FROM {_ET} WHERE dst = n.node_id) as degree
                FROM {_NT} n
                WHERE n.type IN ('class', 'struct', 'interface')
                ORDER BY degree DESC
                LIMIT 5
            """)
            # fmt: on
            hub_results = cur.fetchall()
            insights["hub_classes"] = [
                {"name": r[0], "type": r[1], "file": r[2], "line": r[3], "degree": r[4]}
                for r in hub_results
                if r[4] >= 5
            ][:3]

            # Get most-called symbols (important functions)
            # fmt: off
            cur = conn.execute(  # nosemgrep
                f"""
                SELECT n.name, n.type, n.file, n.line,
                       (SELECT COUNT(*) FROM {_ET} WHERE dst = n.node_id AND type = 'CALLS') as in_calls,
                       (SELECT COUNT(*) FROM {_ET} WHERE src = n.node_id AND type = 'CALLS') as out_calls
                FROM {_NT} n
                WHERE n.type IN ('function', 'method', 'class')
                ORDER BY in_calls DESC
                LIMIT 8
            """)
            # fmt: on
            important_results = cur.fetchall()
            insights["important_symbols"] = [
                {
                    "name": r[0],
                    "type": r[1],
                    "file": r[2],
                    "line": r[3],
                    "in_degree": r[4],
                    "out_degree": r[5],
                    "score": r[4] / max(total_edges, 1),  # Simplified score
                }
                for r in important_results
                if r[4] > 0
            ]
            insights["call_hotspots"] = insights["important_symbols"][:5]

            # Optional richer graph analytics
            try:
                from victor.tools.graph_tool import GraphAnalyzer
                from victor_coding.codebase.graph.registry import create_graph_store

                ga = GraphAnalyzer()
                store = create_graph_store(project_path=str(root))
                nodes = await store.get_all_nodes()
                edges = await store.get_all_edges()
                for n in nodes:
                    ga.add_node(n)
                for e in edges:
                    ga.add_edge(e)

                pagerank_top = ga.pagerank(
                    top_k=8,
                    edge_types=[
                        "CALLS",
                        "REFERENCES",
                        "INHERITS",
                        "IMPLEMENTS",
                        "COMPOSED_OF",
                        "IMPORTS",
                    ],
                )
                centrality_top = ga.degree_centrality(
                    top_k=8,
                    edge_types=[
                        "CALLS",
                        "REFERENCES",
                        "INHERITS",
                        "IMPLEMENTS",
                        "COMPOSED_OF",
                        "IMPORTS",
                    ],
                )

                # Connected components (undirected view)
                visited: Set[str] = set()
                components: List[int] = []
                for node_id in ga.nodes:
                    if node_id in visited:
                        continue
                    stack = [node_id]
                    size = 0
                    while stack:
                        nid = stack.pop()
                        if nid in visited:
                            continue
                        visited.add(nid)
                        size += 1
                        neighbors = [t for t, _et, _w in ga.outgoing.get(nid, [])] + [
                            s for s, _et, _w in ga.incoming.get(nid, [])
                        ]
                        for n2 in neighbors:
                            if n2 not in visited:
                                stack.append(n2)
                    components.append(size)

                components.sort(reverse=True)

                insights["pagerank"] = pagerank_top[:5]
                insights["centrality"] = centrality_top[:5]
                insights["components"] = components[:3]
            except Exception as exc:
                logger.debug(f"Graph analytics fallback skipped: {exc}")

            # Module-level analysis: aggregate to file level
            # Use REFERENCES edges (imports/dependencies) for richer module relationships
            # CALLS edges are sparse as they only track explicit function calls
            # Note: Hidden directories (.*) and archive/ are filtered at index time
            # fmt: off
            cur = conn.execute(  # nosemgrep
                f"""
                SELECT
                    src_n.file as src_module,
                    dst_n.file as dst_module,
                    COUNT(*) as ref_count
                FROM {_ET} e
                JOIN {_NT} src_n ON e.src = src_n.node_id
                JOIN {_NT} dst_n ON e.dst = dst_n.node_id
                WHERE e.type = 'REFERENCES'
                  AND src_n.file != dst_n.file
                  AND src_n.file IS NOT NULL
                  AND dst_n.file IS NOT NULL
                  AND src_n.file NOT LIKE 'tests/%'
                  AND dst_n.file NOT LIKE 'tests/%'
                GROUP BY src_n.file, dst_n.file
                HAVING ref_count >= 2
                """)
            # fmt: on
            module_edges = cur.fetchall()

            if module_edges:
                # Build module adjacency for PageRank calculation
                module_in_degree: Dict[str, int] = defaultdict(int)
                module_out_degree: Dict[str, int] = defaultdict(int)
                module_weighted_in: Dict[str, int] = defaultdict(int)
                all_modules: Set[str] = set()

                for src_mod, dst_mod, count in module_edges:
                    all_modules.add(src_mod)
                    all_modules.add(dst_mod)
                    module_out_degree[src_mod] += 1
                    module_in_degree[dst_mod] += 1
                    module_weighted_in[dst_mod] += count

                # Sort by weighted in-degree (approximation of module importance)
                module_importance = [
                    (
                        mod,
                        module_weighted_in[mod],
                        module_in_degree[mod],
                        module_out_degree[mod],
                    )
                    for mod in all_modules
                ]
                module_importance.sort(key=lambda x: x[1], reverse=True)

                # Classify module roles
                insights["important_modules"] = []
                for mod, weighted_in, in_deg, out_deg in module_importance[:8]:
                    # Determine role
                    if in_deg > out_deg * 2 and in_deg >= 3:
                        role = "service"  # Many callers, few outgoing
                    elif out_deg > in_deg * 2 and out_deg >= 3:
                        role = "orchestrator"  # Calls many modules
                    elif in_deg >= 2 and out_deg >= 2:
                        role = "intermediary"  # Both caller and callee
                    elif in_deg > 0 and out_deg == 0:
                        role = "leaf"  # Terminal module
                    elif out_deg > 0 and in_deg == 0:
                        role = "entry"  # Entry point
                    else:
                        role = "peripheral"

                    insights["important_modules"].append(
                        {
                            "module": mod,
                            "weighted_importance": weighted_in,
                            "in_degree": in_deg,
                            "out_degree": out_deg,
                            "role": role,
                        }
                    )

                # Module coupling detection (high fan-in/fan-out)
                coupling_issues = []
                for mod, _weighted_in, in_deg, out_deg in module_importance:
                    total_degree = in_deg + out_deg
                    if total_degree >= 8:  # High connectivity threshold
                        if in_deg > 5 and out_deg > 5:
                            pattern = "hub"
                        elif in_deg > 5:
                            pattern = "high_fan_in"
                        else:
                            pattern = "high_fan_out"
                        coupling_issues.append(
                            {
                                "module": mod,
                                "pattern": pattern,
                                "in_degree": in_deg,
                                "out_degree": out_deg,
                            }
                        )

                insights["module_coupling"] = coupling_issues[:5]

            # Most-inherited classes (architecture backbone)
            try:
                # fmt: off
                cur = conn.execute(  # nosemgrep
                    f"""
                    SELECT dst_n.name, dst_n.file, dst_n.line, COUNT(*) as subclass_count
                    FROM {_ET} e
                    JOIN {_NT} dst_n ON e.dst = dst_n.node_id
                    WHERE e.type = 'INHERITS' AND dst_n.type = 'class'
                    GROUP BY e.dst
                    ORDER BY subclass_count DESC
                    LIMIT 10
                """)
                # fmt: on
                insights["inheritance_backbone"] = [
                    {"name": r[0], "file": r[1], "line": r[2], "subclasses": r[3]}
                    for r in cur.fetchall()
                    if r[3] >= 3
                ]
            except Exception:
                pass

            # Largest files by symbol count (potential god files)
            try:
                # fmt: off
                cur = conn.execute(  # nosemgrep
                    f"""
                    SELECT file, COUNT(*) as symbol_count
                    FROM {_NT}
                    WHERE type IN ('function', 'class')
                    GROUP BY file
                    ORDER BY symbol_count DESC
                    LIMIT 10
                """)
                # fmt: on
                insights["largest_files"] = [
                    {"file": r[0], "symbols": r[1]} for r in cur.fetchall()
                ]
            except Exception:
                pass

        finally:
            conn.close()

    except Exception as e:
        logger.warning(f"Failed to extract graph insights: {e}")

    return insights


async def generate_enhanced_init_md(
    root_path: Optional[str] = None,
    use_llm: bool = False,
    include_conversations: bool = True,
    on_progress: Optional[callable] = None,
    force: bool = False,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    auto_index: bool = True,
    agent: Optional[Any] = None,
    graph_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate init.md using symbol index, conversation insights, and optional LLM.

    Pipeline: Index → Learn (optional) → LLM enhance (optional)

    When ``graph_context`` is provided (populated by ``victor init --ccg``),
    Step 1 uses the pre-built graph RAG index instead of running
    ``SymbolStore.index_codebase()`` — eliminating the duplicate 35-second
    symbol-parsing pass.

    Args:
        root_path: Root directory to analyze. Defaults to current directory.
        use_llm: Whether to use LLM for enhancement (default: False)
        include_conversations: Whether to include conversation insights (default: True)
        on_progress: Optional callback: fn(stage: str, message: str)
        force: If True, re-index all files ignoring cache.
        auto_index: If True, automatically build code graph index when missing (default: True)
        graph_context: Pre-computed graph statistics from ``victor init --ccg``.
            When provided and ``has_graph`` is True, skips SymbolStore indexing.

    Returns:
        Enhanced init.md content. Falls back gracefully if LLM fails.
    """
    import time

    step_times: dict = {}
    step_start: float = 0

    def progress(stage: str, msg: str, complete: bool = False):
        nonlocal step_start
        if complete and step_start > 0:
            elapsed = time.time() - step_start
            step_times[stage] = elapsed
            if on_progress:
                on_progress(stage, f"✓ {msg} ({elapsed:.1f}s)")
        else:
            step_start = time.time()
            if on_progress:
                on_progress(stage, msg)

    # Step 1: Base content — prefer graph RAG (already indexed) over SymbolStore re-index
    graph_is_ready = bool(graph_context and graph_context.get("has_graph"))
    if graph_is_ready:
        progress("index", "Reading symbols from graph index…")
        base_content = await generate_victor_md_from_graph(
            root_path, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )
        n_nodes = graph_context["stats"].get("total_nodes", 0)  # type: ignore[index]
        progress("index", f"Graph index read ({n_nodes:,} symbols)", complete=True)
    else:
        progress("index", "Building symbol index...")
        base_content = await generate_victor_md_from_index(
            root_path, force=force, include_dirs=include_dirs, exclude_dirs=exclude_dirs
        )
        progress("index", "Symbol index built", complete=True)

    # Step 2: Learn - Add conversation insights
    if include_conversations:
        progress("learn", "Extracting conversation insights...")
        insights = await extract_conversation_insights(root_path)
        sessions = insights.get("session_count", 0)
        progress("learn", f"Insights extracted ({sessions} sessions)", complete=True)

        if sessions > 0:
            enhancements = ["\n## Learned from Conversations\n"]
            enhancements.append(
                f"*Based on {insights['session_count']} sessions, {insights['message_count']} messages*\n"
            )

            if insights.get("hot_files"):
                enhancements.append("### Frequently Referenced Files\n")
                for file_path, count in insights["hot_files"][:8]:
                    enhancements.append(f"- `{file_path}` ({count} references)")
                enhancements.append("")

            if insights.get("common_topics"):
                topics = [t[0] for t in insights["common_topics"][:6]]
                enhancements.append("### Common Topics\n")
                enhancements.append(f"Keywords: {', '.join(topics)}\n")

            if insights.get("faq"):
                enhancements.append("### Frequently Asked Questions\n")
                for faq in insights["faq"][:3]:
                    q = (
                        faq["question"][:100] + "..."
                        if len(faq["question"]) > 100
                        else faq["question"]
                    )
                    enhancements.append(f"- {q}")
                enhancements.append("")

            # Insert before Important Notes
            if "## Important Notes" in base_content:
                parts = base_content.split("## Important Notes")
                base_content = (
                    parts[0] + "\n".join(enhancements) + "\n## Important Notes" + parts[1]
                )
            else:
                base_content += "\n" + "\n".join(enhancements)

    # Step 2.5: Graph - Add graph-based insights (design patterns, important symbols)
    progress("graph", "Analyzing code graph...")
    graph_insights = await extract_graph_insights(root_path)

    # Auto-build index if graph is missing and auto_index is enabled
    if not graph_insights.get("has_graph") and auto_index:
        progress("graph", "No graph data - building index automatically...")
        try:
            from victor_coding.compat.settings import load_settings
            from victor.tools.code_search_tool import _get_or_build_index

            root = Path(root_path).resolve() if root_path else Path.cwd()
            settings = load_settings()
            # Build the index (this populates the graph database)
            index, rebuilt = await _get_or_build_index(root, settings, force_reindex=force)
            status_msg = "Index built" if rebuilt else "Index loaded"
            # Get graph stats if available
            if index.graph_store:
                try:
                    stats = await index.graph_store.stats()
                    node_count = stats.get("nodes", 0)
                    edge_count = stats.get("edges", 0)
                    if node_count or edge_count:
                        status_msg += f" ({node_count} symbols, {edge_count} edges)"
                except Exception:
                    pass
            progress("graph", f"{status_msg}, analyzing...")
            # Re-fetch graph insights after indexing
            graph_insights = await extract_graph_insights(root_path)
        except Exception as e:
            logger.warning(f"Auto-indexing failed: {e}")
            progress("graph", f"Auto-indexing failed: {e}", complete=True)

    if graph_insights.get("has_graph"):
        total_nodes = graph_insights["stats"].get("total_nodes", 0)
        node_types = graph_insights["stats"].get("node_types", {})

        # Build detailed summary with node type breakdown
        summary_parts = [f"{total_nodes:,} nodes"]
        if node_types:
            # Show top 3 node types
            top_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)[:3]
            type_summary = ", ".join([f"{count:,} {name}" for name, count in top_types])
            summary_parts.append(f"({type_summary})")

        progress(
            "graph",
            f"Graph analyzed ({' - '.join(summary_parts)})",
            complete=True,
        )

        graph_section = ["\n## Code Graph Insights\n"]
        graph_section.append(
            f"*{graph_insights['stats'].get('total_nodes', 0)} symbols, {graph_insights['stats'].get('total_edges', 0)} relationships*\n"
        )

        # Design patterns detected
        if graph_insights.get("patterns"):
            graph_section.append("### Detected Design Patterns\n")
            for p in graph_insights["patterns"][:5]:
                details = p.get("details", {})
                if p["pattern"] == "provider_strategy":
                    impls = details.get("implementations", [])
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('base_class', '')}` with {len(impls)} implementations"
                    )
                elif p["pattern"] == "facade":
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` ({details.get('incoming_calls', 0)} callers → {details.get('outgoing_calls', 0)} delegates)"
                    )
                elif p["pattern"] == "composition":
                    composed = details.get("composed_of", [])
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` composed of {len(composed)} components"
                    )
                elif p["pattern"] == "factory":
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', '')}` creates {details.get('creates', 0)} types"
                    )
                else:
                    graph_section.append(
                        f"- **{p['name']}**: `{details.get('class', details.get('base_class', ''))}`"
                    )
            graph_section.append("")

        # Most important symbols (PageRank) — skip Python builtins
        _BUILTIN_NAMES = {
            "len",
            "get",
            "set",
            "append",
            "lower",
            "upper",
            "strip",
            "split",
            "join",
            "format",
            "pop",
            "push",
            "keys",
            "values",
            "items",
            "update",
            "copy",
            "clear",
            "add",
            "remove",
            "sort",
            "extend",
            "insert",
            "count",
            "index",
            "replace",
            "find",
            "startswith",
            "endswith",
            "encode",
            "decode",
            "read",
            "write",
            "close",
            "open",
            "print",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "tuple",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "patch",
        }
        if graph_insights.get("important_symbols"):
            filtered_symbols = [
                sym
                for sym in graph_insights["important_symbols"]
                if sym["name"] not in _BUILTIN_NAMES
            ]
            if filtered_symbols:
                graph_section.append("### Most Important Symbols (PageRank)\n")
                graph_section.append("| Symbol | Type | Connections |")
                graph_section.append("|--------|------|-------------|")
                for sym in filtered_symbols[:6]:
                    conns = f"↓{sym['in_degree']} ↑{sym['out_degree']}"
                    line_ref = f":{sym['line']}" if sym.get("line") else ""
                    location = f"({sym['file']}{line_ref})"
                    graph_section.append(
                        f"| `{sym['name']}` {location} | {sym['type']} | {conns} |"
                    )
                graph_section.append("")

        # Hub classes
        if graph_insights.get("hub_classes"):
            graph_section.append("### Hub Classes (High Connectivity)\n")
            for hub in graph_insights["hub_classes"]:
                # Format: ClassName (file.py:line) for LLM navigation
                line_ref = f":{hub['line']}" if hub.get("line") else ""
                location = f"({hub['file']}{line_ref})"
                graph_section.append(f"- `{hub['name']}` {location} - {hub['degree']} connections")
            graph_section.append("")

        # Module-level architecture (more meaningful than symbol-level for codebase navigation)
        if graph_insights.get("important_modules"):
            graph_section.append("### Key Modules (Architecture)\n")
            graph_section.append("| Module | Role | Connections |")
            graph_section.append("|--------|------|-------------|")
            for mod in graph_insights["important_modules"][:6]:
                role_emoji = {
                    "service": "🔧",
                    "orchestrator": "🎛️",
                    "intermediary": "↔️",
                    "leaf": "🍃",
                    "entry": "🚀",
                    "peripheral": "📦",
                }.get(mod["role"], "")
                conns = f"↓{mod['in_degree']} ↑{mod['out_degree']}"
                graph_section.append(
                    f"| `{mod['module']}` | {role_emoji} {mod['role']} | {conns} |"
                )
            graph_section.append("")

        # Module coupling warnings
        if graph_insights.get("module_coupling"):
            graph_section.append("### Coupling Hotspots\n")
            for coupling in graph_insights["module_coupling"][:3]:
                pattern_desc = {
                    "hub": "⚠️ High fan-in AND fan-out",
                    "high_fan_in": "Many callers",
                    "high_fan_out": "Calls many modules",
                }.get(coupling["pattern"], coupling["pattern"])
                graph_section.append(
                    f"- `{coupling['module']}` - {pattern_desc} "
                    f"(↓{coupling['in_degree']} ↑{coupling['out_degree']})"
                )
            graph_section.append("")

        # Inheritance backbone (most-subclassed types)
        if graph_insights.get("inheritance_backbone"):
            graph_section.append("### Inheritance Backbone\n")
            graph_section.append("| Base Class | Subclasses | Location |")
            graph_section.append("|------------|------------|----------|")
            for cls in graph_insights["inheritance_backbone"][:8]:
                line_ref = f":{cls['line']}" if cls.get("line") else ""
                graph_section.append(
                    f"| `{cls['name']}` | {cls['subclasses']} | {cls['file']}{line_ref} |"
                )
            graph_section.append("")

        # Largest files by symbol count
        if graph_insights.get("largest_files"):
            graph_section.append("### Largest Files (by symbol count)\n")
            for f in graph_insights["largest_files"][:8]:
                graph_section.append(f"- `{f['file']}` — {f['symbols']} symbols")
            graph_section.append("")

        # Insert before Important Notes
        if "## Important Notes" in base_content:
            parts = base_content.split("## Important Notes")
            base_content = parts[0] + "\n".join(graph_section) + "\n## Important Notes" + parts[1]
        else:
            base_content += "\n" + "\n".join(graph_section)
    else:
        # Graph data still missing after potential auto-indexing attempt
        if auto_index:
            progress(
                "graph",
                "Graph indexing incomplete (retry with 'victor index')",
                complete=True,
            )
        else:
            progress(
                "graph",
                "No graph data (run 'victor index' or enable auto_index)",
                complete=True,
            )

    # Step 3: Deep - Use Agent framework to synthesize init.md (full observability)
    if not use_llm:
        return base_content

    progress("deep", "Synthesizing init.md with LLM...")

    try:
        from victor_contracts.init_runtime import InitSynthesizer
        from victor_coding.compat.settings import load_settings as _load

        settings = _load()
        provider_name = getattr(settings, "default_provider", "ollama")
        model_name = getattr(settings, "default_model", None)

        synthesizer = InitSynthesizer()
        synthesized = await synthesizer.synthesize(
            base_content,
            agent=agent,  # Reuse orchestrator from slash command (None for CLI)
            provider=provider_name,
            model=model_name,
        )

        progress("deep", "LLM synthesis complete", complete=True)

        if synthesized:
            return synthesized

        logger.warning("Init synthesis returned empty — using raw base content")
        return base_content

    except Exception as e:
        progress("deep", f"LLM failed: {e}", complete=True)
        logger.warning(f"LLM synthesis failed: {e}, using template fallback")
        return base_content
