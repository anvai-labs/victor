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

"""Dynamic import tracking for Python dependency analysis.

This module extends static graph analysis to detect dynamic import patterns
that are invisible to traditional AST-based import extraction:

1. String-based imports via importlib
2. Plugin registration systems
3. Export mappings in __init__.py
4. Configuration-driven module loading
5. Decorator-based registration
6. Hook methods and protocol implementations

Usage:
    from victor.tools.graph_dynamic_import_tracker import DynamicImportTracker

    tracker = DynamicImportTracker(root_path)
    dynamic_edges = tracker.track_dynamic_imports()
    # Returns: [("caller.py", "dynamically_imported_module.py", "importlib")]
"""

from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DynamicImport:
    """Represents a dynamically detected import relationship."""

    source_file: str
    target_module: str
    import_type: str  # "importlib", "__init__", "plugin", "decorator", "config"
    line_number: Optional[int] = None
    confidence: float = 1.0  # 0.0-1.0, lower means more uncertain
    context: str = ""  # Surrounding code context for verification


@dataclass
class ExportMapping:
    """Represents an export from __init__.py."""

    init_file: str
    exported_name: str
    target_module: str  # The actual module being re-exported
    export_type: str  # "direct", "conditional", "alias"


class DynamicImportTracker:
    """Tracks dynamic imports and export mappings in Python projects.

    Uses AST parsing and pattern matching to detect dynamic import patterns
    that traditional static analysis misses.
    """

    # Patterns that indicate dynamic module loading
    IMPORTLIB_PATTERNS = [
        r'importlib\.import_module\s*\(\s*["\']([^"\']+)["\']',
        r'__import__\s*\(\s*["\']([^"\']+)["\']',
        r'import_module\s*\(\s*["\']([^"\']+)["\']',
        r'load_module\s*\(\s*["\']([^"\']+)["\']',
    ]

    # Method names that typically return module paths for dynamic loading
    HOOK_METHOD_PATTERNS = [
        r"_get_\w+_module\s*\(\s*\)\s*->\s*str",
        r"get_\w+_module\s*\(\s*\)\s*str",
        r"_get_\w+_path\s*\(\s*\)\s*str",
        r"module_path\s*\(\s*\)\s*str",
    ]

    # Path hints that suggest dynamic entry points
    DYNAMIC_ENTRYPOINT_PATHS = {
        "/plugins/",
        "/languages/plugins/",
        "/workflows/",
        "/integrations/",
        "/adapters/",
        "/providers/",
        "/handlers/",
        "/escape_hatches",
        "/extensions/",
    }

    # Decorators that indicate registration
    REGISTRATION_DECORATORS = {
        "@register",
        "@register_plugin",
        "@register_handler",
        "@hook",
        "@plugin",
        "@provider",
    }

    def __init__(self, root_path: Path | str):
        """Initialize the tracker.

        Args:
            root_path: Root directory of the project to analyze
        """
        self.root_path = Path(root_path).resolve()
        self._init_exports: Dict[str, List[ExportMapping]] = {}
        self._plugin_files: Set[str] = set()
        self._hook_methods: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    def scan_all(self) -> List[DynamicImport]:
        """Scan all Python files for dynamic imports.

        Returns:
            List of detected dynamic imports
        """
        all_imports = []

        # First pass: find all __init__.py exports and plugin files
        self._scan_init_files()
        self._identify_plugin_files()

        # Second pass: find dynamic imports in all files
        for py_file in self.root_path.rglob("*.py"):
            try:
                imports = self._scan_file_for_dynamic_imports(py_file)
                all_imports.extend(imports)
            except Exception as e:
                logger.debug(f"Error scanning {py_file}: {e}")

        return all_imports

    def _scan_init_files(self) -> None:
        """Scan all __init__.py files for export mappings."""
        for init_file in self.root_path.rglob("__init__.py"):
            try:
                exports = self._parse_init_exports(init_file)
                self._init_exports[str(init_file.relative_to(self.root_path))] = exports
            except Exception as e:
                logger.debug(f"Error parsing {init_file}: {e}")

    def _parse_init_exports(self, init_file: Path) -> List[ExportMapping]:
        """Parse export patterns from __init__.py."""
        try:
            source = init_file.read_text(encoding="utf-8")
        except Exception:
            return []

        try:
            tree = ast.parse(source, filename=str(init_file))
        except SyntaxError:
            return []

        class InitVisitor(ast.NodeVisitor):
            def __init__(self, init_path: str) -> None:
                self.init_path = init_path
                self.exports: List[ExportMapping] = []

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    for alias in node.names:
                        self.exports.append(
                            ExportMapping(
                                init_file=self.init_path,
                                exported_name=alias.asname or alias.name,
                                target_module=(
                                    f"{node.module}.{alias.name}"
                                    if alias.name != "*"
                                    else node.module
                                ),
                                export_type="direct",
                            )
                        )
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                # Look for __all__ exports
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id == "__all__":
                        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    self.exports.append(
                                        ExportMapping(
                                            init_file=self.init_path,
                                            exported_name=elt.value,
                                            target_module=f".{elt.value}",  # Relative import
                                            export_type="__all__",
                                        )
                                    )
                self.generic_visit(node)

            def visit_If(self, node: ast.If) -> None:
                # Check for conditional exports (TYPE_CHECKING pattern)
                for test_node in ast.walk(node.test):
                    if isinstance(test_node, ast.Name) and test_node.id == "TYPE_CHECKING":
                        for body_node in ast.walk(node):
                            if isinstance(body_node, ast.ImportFrom):
                                for alias in body_node.names:
                                    self.exports.append(
                                        ExportMapping(
                                            init_file=self.init_path,
                                            exported_name=alias.asname or alias.name,
                                            target_module=(
                                                f"{body_node.module}.{alias.name}"
                                                if body_node.module
                                                else alias.name
                                            ),
                                            export_type="conditional",
                                        )
                                    )
                                break
                        break
                self.generic_visit(node)

        visitor = InitVisitor(str(init_file.relative_to(self.root_path)))
        visitor.visit(tree)
        return visitor.exports

    def _identify_plugin_files(self) -> None:
        """Identify files in plugin directories."""
        for path_hint in self.DYNAMIC_ENTRYPOINT_PATHS:
            plugin_dir = self.root_path
            for part in path_hint.split("/"):
                plugin_dir = plugin_dir / part.lstrip("/")
                if not plugin_dir.exists():
                    break

            if plugin_dir.exists() and plugin_dir.is_dir():
                for py_file in plugin_dir.rglob("*.py"):
                    if py_file.name != "__init__.py":
                        self._plugin_files.add(str(py_file.relative_to(self.root_path)))

    def _scan_file_for_dynamic_imports(self, file_path: Path) -> List[DynamicImport]:
        """Scan a single file for dynamic import patterns."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        imports = []
        rel_path = str(file_path.relative_to(self.root_path))

        # Check for importlib patterns
        for pattern in self.IMPORTLIB_PATTERNS:
            for match in re.finditer(pattern, source):
                module_name = match.group(1)
                line_num = source[: match.start()].count("\n") + 1
                imports.append(
                    DynamicImport(
                        source_file=rel_path,
                        target_module=module_name,
                        import_type="importlib",
                        line_number=line_num,
                        confidence=0.9,
                    )
                )

        # AST-based detection
        try:
            tree = ast.parse(source, filename=str(file_path))
            imports.extend(self._ast_scan_dynamic_imports(tree, rel_path, source))
        except SyntaxError:
            pass

        return imports

    def _ast_scan_dynamic_imports(
        self, tree: ast.AST, file_path: str, source: str
    ) -> List[DynamicImport]:
        """Scan AST for dynamic import patterns."""

        class DynamicImportVisitor(ast.NodeVisitor):
            def __init__(self, path: str, src: str) -> None:
                self.path = path
                self.src = src
                self.imports: List[DynamicImport] = []

            def visit_Call(self, node: ast.Call) -> None:
                # Check for importlib.import_module calls
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "import_module":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            if isinstance(node.args[0].value, str):
                                self.imports.append(
                                    DynamicImport(
                                        source_file=self.path,
                                        target_module=node.args[0].value,
                                        import_type="importlib",
                                        line_number=node.lineno,
                                        confidence=0.95,
                                    )
                                )

                # Check for __import__ calls
                elif isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if isinstance(node.args[0].value, str):
                            self.imports.append(
                                DynamicImport(
                                    source_file=self.path,
                                    target_module=node.args[0].value,
                                    import_type="__import__",
                                    line_number=node.lineno,
                                    confidence=0.9,
                                )
                            )

                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # Check for hook methods that return module paths
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if decorator.id in {"property", "staticmethod"}:
                            break

                # Check return annotation for module path hints
                if node.returns and isinstance(node.returns, ast.Constant):
                    if (
                        isinstance(node.returns.value, str)
                        and "module" in node.returns.value.lower()
                    ):
                        # Scan function body for string returns
                        for body_node in ast.walk(node):
                            if isinstance(body_node, ast.Return):
                                if isinstance(body_node.value, ast.Constant):
                                    if isinstance(body_node.value.value, str):
                                        self.imports.append(
                                            DynamicImport(
                                                source_file=self.path,
                                                target_module=body_node.value.value,
                                                import_type="hook_method",
                                                line_number=node.lineno,
                                                confidence=0.7,
                                            )
                                        )

                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                # Check for plugin/protocol base classes
                if node.bases:
                    for base in node.bases:
                        base_name = None
                        if isinstance(base, ast.Name):
                            base_name = base.id
                        elif isinstance(base, ast.Attribute):
                            base_name = base.attr

                        if base_name and any(
                            keyword in base_name.lower()
                            for keyword in [
                                "plugin",
                                "handler",
                                "provider",
                                "protocol",
                                "hook",
                            ]
                        ):
                            # This class is likely dynamically registered
                            self.imports.append(
                                DynamicImport(
                                    source_file=self.path,
                                    target_module=node.name,
                                    import_type="plugin_class",
                                    line_number=node.lineno,
                                    confidence=0.6,
                                )
                            )
                self.generic_visit(node)

        visitor = DynamicImportVisitor(file_path, source)
        visitor.visit(tree)
        return visitor.imports

    def get_dynamic_dependencies(self, file_path: str) -> List[str]:
        """Get all dynamic dependencies for a specific file.

        Args:
            file_path: Relative path to the file

        Returns:
            List of module names that are dynamically imported
        """
        deps = set()

        # Direct dynamic imports
        for imp in self.scan_all():
            if imp.source_file == file_path:
                deps.add(imp.target_module)

        # Indirect via __init__ exports
        for init_file, exports in self._init_exports.items():
            for export in exports:
                if export.target_module in file_path or file_path.endswith(export.target_module):
                    deps.add(export.exported_name)

        return sorted(deps)

    def get_reverse_dynamic_dependencies(self, module_name: str) -> List[str]:
        """Find all files that dynamically import the given module.

        Args:
            module_name: Name of the module to look up

        Returns:
            List of files that dynamically import this module
        """
        importers = []
        normalized = module_name.replace("/", ".").replace(".py", "")

        for imp in self.scan_all():
            if normalized in imp.target_module or imp.target_module in normalized:
                importers.append(imp.source_file)

        # Check __init__ exports
        for init_file, exports in self._init_exports.items():
            for export in exports:
                if export.target_module == normalized or export.target_module.startswith(
                    f"{normalized}."
                ):
                    importers.append(init_file)

        return sorted(set(importers))

    def is_dynamic_entrypoint(self, file_path: str) -> Tuple[bool, str]:
        """Check if a file is a dynamic entry point.

        Args:
            file_path: Relative path to check

        Returns:
            Tuple of (is_entrypoint, reason)
        """
        # Check if in plugin directory
        for path_hint in self.DYNAMIC_ENTRYPOINT_PATHS:
            if path_hint.replace("/", "") in file_path.replace("/", ""):
                return True, f"in_plugin_directory:{path_hint}"

        # Check if exported from __init__.py
        for init_file, exports in self._init_exports.items():
            for export in exports:
                if file_path.endswith(export.target_module) or export.target_module in file_path:
                    return True, f"exported_from:{init_file}"

        # Check if has hook methods
        if file_path in self._hook_methods:
            return True, f"has_hook_methods:{len(self._hook_methods[file_path])}"

        return False, ""

    def augment_graph_analysis(self, static_callers: Set[str], symbol_name: str) -> Dict[str, Any]:
        """Augment static graph analysis with dynamic import information.

        Args:
            static_callers: Set of files that statically reference the symbol
            symbol_name: Name of the symbol being analyzed

        Returns:
            Dictionary with augmented analysis including dynamic references
        """
        dynamic_importers = set()
        export_locations = []

        # Find files that dynamically import this module
        for imp in self.scan_all():
            if symbol_name in imp.target_module or imp.target_module.endswith(symbol_name):
                dynamic_importers.add(imp.source_file)

        # Find __init__.py exports
        for init_file, exports in self._init_exports.items():
            for export in exports:
                if export.exported_name == symbol_name or export.target_module.endswith(
                    symbol_name
                ):
                    export_locations.append(
                        {
                            "init_file": init_file,
                            "exported_as": export.exported_name,
                            "target": export.target_module,
                            "type": export.export_type,
                        }
                    )

        return {
            "static_callers": list(static_callers),
            "dynamic_importers": list(dynamic_importers - static_callers),
            "exported_from": export_locations,
            "is_dynamic_entrypoint": self.is_dynamic_entrypoint(symbol_name),
            "total_references": len(static_callers | dynamic_importers),
        }


def check_dynamic_imports_for_file(file_path: str | Path, root_path: str | Path) -> Dict[str, Any]:
    """Convenience function to check dynamic imports for a single file.

    Args:
        file_path: Path to the file to check
        root_path: Root directory of the project

    Returns:
        Dictionary with dynamic import analysis
    """
    tracker = DynamicImportTracker(root_path)
    rel_path = str(Path(file_path).relative_to(Path(root_path)))

    return {
        "file": rel_path,
        "dynamic_imports": [
            imp.target_module for imp in tracker.scan_all() if imp.source_file == rel_path
        ],
        "dynamically_imported_by": tracker.get_reverse_dynamic_dependencies(rel_path),
        "is_entrypoint": tracker.is_dynamic_entrypoint(rel_path),
    }
