"""Manifest validation for external verticals.

Provides static analysis of a vertical class's manifest for completeness
and correctness. Zero dependency on victor-ai — pure SDK.

Usage::

    from victor_contracts.verticals.validation import validate_manifest

    report = validate_manifest(MyVertical)
    if not report.ok:
        for issue in report.issues:
            print(f"[{issue.level}] {issue.code}: {issue.message}")
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Optional, Set, Type

from victor_contracts.core.api_version import (
    CURRENT_API_VERSION,
    MIN_SUPPORTED_API_VERSION,
)
from victor_contracts.validation import ValidationReport
from victor_contracts.verticals.manifest import ExtensionManifest


def validate_manifest(vertical_cls: Type[Any]) -> ValidationReport:
    """Validate a vertical's manifest for completeness and correctness.

    Checks:
    - Manifest presence (_victor_manifest attribute or get_manifest())
    - Name non-empty
    - API version within supported range
    - Required VerticalBase methods implemented

    Args:
        vertical_cls: The vertical class to validate.

    Returns:
        ValidationReport with issues (errors and warnings).
    """
    name = getattr(vertical_cls, "__name__", str(vertical_cls))
    report = ValidationReport(package_name=name)

    # 1. Check manifest presence
    manifest = _get_manifest(vertical_cls)
    if manifest is None:
        report.add_issue(
            "missing_manifest",
            f"Class {name} has no _victor_manifest attribute and get_manifest() "
            f"returned None. Use @register_vertical() or set _victor_manifest.",
        )
        return report

    # 2. Check name
    if not manifest.name or not manifest.name.strip():
        report.add_issue("empty_manifest_name", "Manifest name is empty.")

    # 3. Check API version
    if manifest.api_version < MIN_SUPPORTED_API_VERSION:
        report.add_issue(
            "api_version_unsupported",
            f"API version {manifest.api_version} is below minimum "
            f"supported ({MIN_SUPPORTED_API_VERSION}).",
        )
    elif manifest.api_version > CURRENT_API_VERSION:
        report.add_issue(
            "api_version_future",
            f"API version {manifest.api_version} is above current "
            f"({CURRENT_API_VERSION}). This vertical may require a newer framework.",
            level="warning",
        )

    # 4. Check required methods
    for method_name in ("get_name", "get_description", "get_tools"):
        method = getattr(vertical_cls, method_name, None)
        if method is None:
            report.add_issue(
                f"missing_method:{method_name}",
                f"Required method {method_name}() not found on {name}.",
            )

    return report


def audit_vertical_dependencies(
    source_dir: str,
    manifest: Optional[ExtensionManifest] = None,
) -> ValidationReport:
    """Compare actual imports in a vertical's source against declared deps.

    Scans Python files in source_dir, extracts third-party import names,
    and compares against manifest.extension_dependencies.

    Args:
        source_dir: Path to the vertical's source directory.
        manifest: The vertical's ExtensionManifest (optional).

    Returns:
        ValidationReport with warnings for undeclared dependencies.
    """
    report = ValidationReport(package_name=source_dir)
    source_path = Path(source_dir)

    if not source_path.exists():
        report.add_issue(
            "source_not_found",
            f"Source directory not found: {source_dir}",
        )
        return report

    # Collect all imports
    all_imports: Set[str] = set()
    for py_file in source_path.rglob("*.py"):
        all_imports.update(_extract_top_level_imports(py_file))

    # Filter to third-party only
    stdlib = _get_stdlib_modules()
    third_party = {
        imp
        for imp in all_imports
        if imp not in stdlib
        and not imp.startswith("victor_contracts")
        and not imp.startswith("victor_contracts")
        and not imp.startswith("victor.")
        and not imp.startswith(".")
    }

    # Compare against declared extension_dependencies
    if manifest is not None:
        declared = {dep.extension_name for dep in manifest.extension_dependencies}
        undeclared = third_party - declared
        for dep in sorted(undeclared):
            report.add_issue(
                f"undeclared_dependency:{dep}",
                f"Import '{dep}' found in source but not declared in "
                f"manifest.extension_dependencies.",
                level="warning",
            )

    return report


def _get_manifest(vertical_cls: Type[Any]) -> Optional[ExtensionManifest]:
    """Extract manifest from a vertical class."""
    manifest: object = getattr(vertical_cls, "_victor_manifest", None)
    if manifest is not None and isinstance(manifest, ExtensionManifest):
        return manifest
    get_fn = getattr(vertical_cls, "get_manifest", None)
    if callable(get_fn):
        try:
            result = get_fn()
            if isinstance(result, ExtensionManifest):
                return result
        except Exception:
            pass
    return None


def _extract_top_level_imports(filepath: Path) -> Set[str]:
    """Extract top-level module names from import statements."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
    return modules


def _get_stdlib_modules() -> Set[str] | frozenset[str]:
    """Get standard library module names."""
    if hasattr(sys, "stdlib_module_names"):
        return sys.stdlib_module_names  # Python 3.10+
    # Fallback for older Python
    return {
        "abc",
        "ast",
        "asyncio",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "hashlib",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "random",
        "re",
        "shutil",
        "signal",
        "socket",
        "sqlite3",
        "string",
        "subprocess",
        "sys",
        "tempfile",
        "textwrap",
        "threading",
        "time",
        "traceback",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "weakref",
        "xml",
    }
