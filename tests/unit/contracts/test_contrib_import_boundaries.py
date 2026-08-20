# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression guard: contrib verticals must not import from victor internals.

The contrib directory (victor/verticals/contrib/) is deprecated — all verticals
are extracted to external packages (victor-coding, victor-devops, etc.).
This test ensures the tombstone directory doesn't accumulate non-SDK imports
during refactoring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Forbidden import prefixes — verticals must NOT import these
FORBIDDEN_PREFIXES = (
    "victor.agent",
    "victor.core.container",
    "victor.core.bootstrap",
    "victor.core.database",
    "victor.providers",
    "victor.evaluation",
)

# Allowed imports (SDK + framework shims for backward compat)
ALLOWED_PREFIXES = (
    "victor_contracts",
    "victor.framework.vertical_base",  # Shim that re-exports SDK
    "victor.framework.extensions",  # Shim that re-exports SDK
)


def _get_contrib_dir() -> Path:
    """Locate the contrib verticals directory."""
    repo_root = Path(__file__).parent.parent.parent.parent
    return repo_root / "victor" / "verticals" / "contrib"


def _extract_imports(filepath: Path) -> list:
    """Extract all import module names from a Python file via AST."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


class TestContribImportBoundaries:
    """Guard: contrib directory must not regress with internal imports."""

    def test_contrib_directory_removed(self):
        """Contrib directory was removed — all verticals extracted to external packages."""
        contrib = _get_contrib_dir()
        assert not contrib.exists(), (
            f"Contrib directory still exists at {contrib}. "
            f"It was removed in v0.7.0 — all verticals are external packages now."
        )

    def test_contrib_files_have_no_forbidden_imports(self):
        contrib = _get_contrib_dir()
        violations = []

        for py_file in contrib.rglob("*.py"):
            imports = _extract_imports(py_file)
            for imp in imports:
                is_forbidden = any(imp.startswith(p) for p in FORBIDDEN_PREFIXES)
                is_allowed = any(imp.startswith(p) for p in ALLOWED_PREFIXES)
                if is_forbidden and not is_allowed:
                    violations.append(f"{py_file.relative_to(contrib)}:{imp}")

        assert (
            not violations
        ), "Contrib verticals must not import from victor internals. " "Violations:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

    def test_contrib_init_is_tombstone(self):
        """The contrib __init__.py should be empty or a deprecation stub."""
        init_file = _get_contrib_dir() / "__init__.py"
        if not init_file.exists():
            pytest.skip("No contrib __init__.py")

        imports = _extract_imports(init_file)
        forbidden = [
            imp
            for imp in imports
            if any(imp.startswith(p) for p in FORBIDDEN_PREFIXES)
            and not any(imp.startswith(p) for p in ALLOWED_PREFIXES)
        ]
        assert not forbidden, f"contrib/__init__.py has forbidden imports: {forbidden}"
