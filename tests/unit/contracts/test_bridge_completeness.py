# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Bridge completeness: verify every ``from victor_contracts.* import`` in the
victor-coding vertical resolves through the lazy-import bridge.

Prevents runtime ImportErrors like ``resolve_project_db_root`` (#349) and
``GraphManager`` (#351) — gaps where the vertical imports a name from
``victor_contracts.*`` but the bridge module's ``_LAZY_IMPORTS`` dict doesn't
proxy it. Each such gap is a guaranteed crash on every graph/code_search call
until the bridge is patched.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# The vertical source tree to audit.
_VERTICAL_ROOT = (
    Path(__file__).resolve().parents[3] / "verticals" / "victor-coding" / "victor_coding"
)

# Bridge modules that might be imported by the vertical.
_CONTRACTS_PREFIX = "victor_contracts"


def _find_contracts_imports(root: Path) -> list[tuple[str, str, str, int]]:
    """AST-scan ``root`` for ``from victor_contracts.X import Y`` statements.

    Returns ``[(file, module, name, line), ...]``.
    """
    results: list[tuple[str, str, str, int]] = []
    if not root.exists():
        return results
    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(_CONTRACTS_PREFIX)
            ):
                for alias in node.names:
                    results.append((str(py_file), node.module, alias.name, node.lineno))
    return results


def _name_is_bridged(module_name: str, attr_name: str) -> bool:
    """Check that ``module_name.attr_name`` resolves through the bridge."""
    try:
        mod = importlib.import_module(module_name)
        getattr(mod, attr_name)
        return True
    except Exception:
        return False


def test_all_victor_coding_imports_are_bridged():
    """Every name imported from victor_contracts.* by the victor-coding vertical
    must resolve through the bridge. Prevents runtime ImportErrors like
    resolve_project_db_root (#349) and GraphManager (#351).
    """
    imports = _find_contracts_imports(_VERTICAL_ROOT)
    assert imports, "Expected to find victor_contracts imports in victor-coding"

    gaps: list[str] = []
    for file_path, module_name, attr_name, line_no in imports:
        if not _name_is_bridged(module_name, attr_name):
            rel = Path(file_path).relative_to(_VERTICAL_ROOT.parent.parent)
            gaps.append(f"  {rel}:{line_no} — {module_name}.{attr_name} not bridged")

    if gaps:
        pytest.fail(
            f"{len(gaps)} victor_contracts import(s) are NOT bridged "
            f"(will crash at runtime):\n" + "\n".join(gaps)
        )


def test_graph_tool_module_loads():
    """The graph tool module must always be importable, even if a bridge
    entry is missing. Module-level imports must be hardened."""
    pytest.importorskip("victor_coding")  # vertical not installed in CI's Quick-Unit env
    try:
        importlib.import_module("victor_coding.tools.graph_tool")
    except ImportError as exc:
        pytest.fail(f"graph_tool.py failed to import: {exc}")
