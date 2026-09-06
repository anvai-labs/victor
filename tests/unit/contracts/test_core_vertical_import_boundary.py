"""Guard tests: victor/ (core) must NOT import from external vertical packages.

This is the reverse of test_external_vertical_import_boundaries.py which checks
that verticals don't import from core. This file checks that core doesn't import
from verticals — even in try/except blocks or via importlib.import_module().

External vertical packages should be discovered via entry points and the
CapabilityRegistry, not imported by name.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from victor_contracts.testing.boundaries import KNOWN_VERTICAL_PACKAGE_NAMES

# Root of the victor core package (not victor-contracts, not external verticals)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VICTOR_ROOT = REPO_ROOT / "victor"

# External vertical package prefixes that core must never import. Sourced from
# the shared manifest (co-design review item 22b) rather than a locally
# maintained list.
EXTERNAL_VERTICAL_PREFIXES = KNOWN_VERTICAL_PACKAGE_NAMES

# Known violations to track migration progress (baseline).
# These are architectural violations that should be migrated to entry points.
#
# Empty: the last tracked violation (victor/core/graph_rag/language_handlers.py
# importing victor_coding.languages.registry as a legacy edge-handler fallback)
# was removed. Edge handlers now resolve exclusively through the
# TreeSitterAnalysisProtocol capability, whose registry access lazily runs
# plugin bootstrap — so any installed language provider registers itself
# without core importing it by name. Core is now fully decoupled from external
# verticals; keep this empty (test_no_stale_known_violations enforces it).
KNOWN_VIOLATIONS: Dict[str, Set[str]] = {}

# Files in root that must NEVER import victor_coding directly, even via the
# KNOWN_VIOLATIONS allowlist. These are the canonical tree-sitter consumers
# that the TSA migration explicitly requires to go through
# TreeSitterAnalysisProtocol.
TREE_SITTER_CONSUMER_FILES = (
    "victor/core/graph_rag/indexing.py",
    "victor/core/indexing/ccg_builder.py",
)


def _collect_all_imports(root: Path) -> List[Tuple[str, int, str]]:
    """AST-walk all .py files under root and collect import statements.

    Returns list of (relative_path, line_number, module_name) for imports
    that match external vertical prefixes. Only collects actual ast.Import
    and ast.ImportFrom nodes — NOT string literals in docstrings.
    """
    violations = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            rel = str(py_file.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(py_file.relative_to(root))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if any(node.module.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                    violations.append((rel, node.lineno, f"from {node.module}"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                        violations.append((rel, node.lineno, f"import {alias.name}"))
    return violations


def _collect_dynamic_import_calls(root: Path) -> List[Tuple[str, int, str]]:
    """Find dynamic imports of external vertical packages in AST.

    Returns list of (relative_path, line_number, module_string) for dynamic
    imports of external vertical packages.
    """
    violations = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            rel = str(py_file.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(py_file.relative_to(root))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match importlib.import_module(...), import_module(...), or __import__(...)
            func = node.func
            is_dynamic_import = False
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                is_dynamic_import = True
            elif isinstance(func, ast.Name) and func.id in {
                "import_module",
                "__import__",
            }:
                is_dynamic_import = True
            if not is_dynamic_import:
                continue
            # Check first positional argument is a string literal
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                mod_name = node.args[0].value
                if any(mod_name.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                    violations.append((rel, node.lineno, f'dynamic_import("{mod_name}")'))
    return violations


class TestCoreDoesNotImportExternalVerticals:
    """Ensure victor/ has zero static or dynamic imports from external verticals."""

    def test_external_vertical_prefixes_has_not_shrunk(self) -> None:
        """Regression guard (co-design review item 22b): EXTERNAL_VERTICAL_PREFIXES
        is sourced from victor_contracts.testing.boundaries.KNOWN_VERTICAL_PACKAGE_NAMES.
        Pin the exact expected set so a future edit to the shared manifest can't
        silently shrink this guard's scan target without a visible, deliberate
        test change here.
        """
        assert set(EXTERNAL_VERTICAL_PREFIXES) == {
            "victor_coding",
            "victor_devops",
            "victor_research",
            "victor_rag",
            "victor_dataanalysis",
            "victor_invest",
        }

    def test_dynamic_import_collector_flags_import_module_and_dunder_import(
        self,
        tmp_path: Path,
    ):
        package_dir = tmp_path / "victor"
        package_dir.mkdir()
        (package_dir / "sample.py").write_text(
            "\n".join(
                [
                    "import importlib",
                    "importlib.import_module('victor_coding.plugin')",
                    "__import__('victor_research.plugin')",
                ]
            ),
            encoding="utf-8",
        )

        violations = _collect_dynamic_import_calls(package_dir)

        assert violations == [
            ("sample.py", 2, 'dynamic_import("victor_coding.plugin")'),
            ("sample.py", 3, 'dynamic_import("victor_research.plugin")'),
        ]

    def test_no_static_imports_of_external_verticals_in_core(self):
        """AST-walk all .py under victor/ and assert zero import statements
        referencing external vertical packages (even try/except guarded ones).

        Docstring occurrences (string literals) are NOT flagged — only actual
        ast.Import / ast.ImportFrom nodes.
        """
        violations = _collect_all_imports(VICTOR_ROOT)

        # Filter out known violations (tracked migration debt)
        known_violations: Set[str] = set()
        for package_violations in KNOWN_VIOLATIONS.values():
            known_violations.update(package_violations)

        new_violations = [
            (f, line, mod)
            for f, line, mod in violations
            if not any(
                # Match both "from victor_coding..." and "import victor_coding..."
                mod == known or mod.startswith(f"from {known}") or mod.startswith(f"import {known}")
                for known in known_violations
            )
        ]

        assert not new_violations, (
            f"Core (victor/) has {len(new_violations)} static import(s) from external "
            f"verticals. Use entry points or CapabilityRegistry instead:\n"
            + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in new_violations)
        )

    def test_no_dynamic_imports_of_external_verticals(self):
        """Find dynamic import patterns for external vertical packages.

        Dynamic imports by package name are still coupling — use
        CapabilityRegistry or entry point discovery instead.
        """
        violations = _collect_dynamic_import_calls(VICTOR_ROOT)
        assert not violations, (
            f"Core (victor/) has {len(violations)} dynamic import(s) of external "
            f"verticals. Use CapabilityRegistry instead:\n"
            + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in violations)
        )

    def test_no_direct_tree_sitter_imports_in_root_extraction_paths(self):
        """Specific tree-sitter consumer files must never import victor_coding.

        These files are part of the TSA migration path and should always go
        through TreeSitterAnalysisProtocol via the capability registry. The
        global KNOWN_VIOLATIONS allowlist does NOT apply here — these paths
        get a stricter guarantee.
        """
        offenders: List[Tuple[str, int, str]] = []
        for rel in TREE_SITTER_CONSUMER_FILES:
            path = REPO_ROOT / rel
            assert path.exists(), f"Guarded file disappeared: {rel}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if any(node.module.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                        offenders.append((rel, node.lineno, f"from {node.module}"))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name.startswith(p) for p in EXTERNAL_VERTICAL_PREFIXES):
                            offenders.append((rel, node.lineno, f"import {alias.name}"))
        assert not offenders, (
            "Tree-sitter consumer files must use TreeSitterAnalysisProtocol "
            "instead of importing victor_coding directly:\n"
            + "\n".join(f"  {f}:{line}: {mod}" for f, line, mod in offenders)
        )

    def test_known_violations_not_stale(self):
        """Ensure KNOWN_VIOLATIONS baseline stays current (no stale entries).

        If a violation is fixed, remove it from KNOWN_VIOLATIONS.
        """
        violations = _collect_all_imports(VICTOR_ROOT)

        # Collect all actual violation import prefixes (strip "from " and "import " prefixes)
        actual_violation_prefixes: Set[str] = set()
        for _, _, mod in violations:
            # Strip "from " or "import " prefix to get the raw module name
            if mod.startswith("from "):
                actual_violation_prefixes.add(
                    mod[5:].split()[0]
                )  # Remove "from " and take first word
            elif mod.startswith("import "):
                actual_violation_prefixes.add(mod[7:])  # Remove "import "
            else:
                actual_violation_prefixes.add(mod)

        # Check for stale entries (violations that are fixed but still in baseline)
        for package, known_imports in KNOWN_VIOLATIONS.items():
            stale = {
                imp
                for imp in known_imports
                if not any(
                    actual == imp or actual.startswith(f"{imp}.") or imp.startswith(f"{actual}.")
                    for actual in actual_violation_prefixes
                )
            }
            if stale:
                pytest.fail(
                    f"Package {package} has stale KNOWN_VIOLATIONS entries (violations fixed!):\n"
                    f"  {stale}\n"
                    f"Remove these from KNOWN_VIOLATIONS in test_core_vertical_import_boundary.py"
                )
