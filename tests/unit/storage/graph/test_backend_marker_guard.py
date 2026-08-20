# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Guard: production code must resolve the graph backend through ``"auto"``.

``create_graph_store("auto", …)`` honors ``<project>/.victor/graph_backend`` and
falls back to SQLite. Hardcoding ``"sqlite"`` in a *write* path while the read
paths resolve through ``"auto"`` produces split-brain: a repo that sets the
marker has its queries routed to a store nothing ever populated.

That is exactly what happened — ``victor init`` built the index into SQLite while
``graph_query_tool`` and ``graph_manager`` read from the marker's backend, so the
per-repo flag documented in ``docs/architecture/proximadb-codegraph-backend.md``
silently did nothing for the command that matters most.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
VICTOR = REPO_ROOT / "victor"

# Modules allowed to name a concrete backend: the factory itself implements the
# mapping, and the parity/bench harnesses deliberately pin one backend to compare
# them. Everything else must go through "auto".
ALLOWED = {
    Path("victor/storage/graph/registry.py"),
    Path("victor/storage/graph/__init__.py"),  # docstring example only
}


def _hardcoded_backend_calls() -> List[Tuple[str, int, str]]:
    """Return (relpath, lineno, literal) for create_graph_store calls not using auto."""
    offenders: List[Tuple[str, int, str]] = []
    for path in VICTOR.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel in ALLOWED:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "create_graph_store":
                continue

            literal = None
            if node.args and isinstance(node.args[0], ast.Constant):
                literal = node.args[0].value
            for kw in node.keywords:
                if kw.arg in {"name", "backend"} and isinstance(kw.value, ast.Constant):
                    literal = kw.value.value

            if isinstance(literal, str) and literal.lower() != "auto":
                offenders.append((str(rel), node.lineno, literal))
    return offenders


def test_production_code_resolves_backend_through_auto():
    offenders = _hardcoded_backend_calls()
    assert not offenders, (
        "create_graph_store() must be called with 'auto' so the per-repo "
        "<project>/.victor/graph_backend marker is honored. Hardcoding a backend "
        "in a write path while read paths use 'auto' yields split-brain:\n"
        + "\n".join(f"  {rel}:{line} -> {lit!r}" for rel, line, lit in offenders)
    )


def test_create_graph_store_takes_at_most_two_positional_args():
    """Regression: a 3-positional call raised TypeError inside a try/except.

    ``project_context.py`` called ``create_graph_store(name, db_path, root)``.
    The factory takes ``(name, project_path)`` only, so every call raised
    TypeError — swallowed by the surrounding ``except Exception``, which made the
    entire graph-context section dead code that never ran once.
    """
    import inspect

    from victor.storage.graph import create_graph_store

    params = inspect.signature(create_graph_store).parameters
    positional = [
        p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) == 2, (
        "create_graph_store's positional arity changed; callers passing a third "
        "positional argument fail at runtime and are easy to hide in a try/except."
    )

    offenders = []
    for path in VICTOR.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name == "create_graph_store" and len(node.args) > 2:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, "create_graph_store called with >2 positional args: " + ", ".join(
        offenders
    )
