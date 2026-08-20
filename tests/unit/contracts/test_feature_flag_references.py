# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CI guard: every referenced FeatureFlag must still exist in the enum.

Prevents regressions where a flag is removed from the enum while code elsewhere
in the monorepo (notably the vendored verticals under ``verticals/`` and
``victor-contracts/``, which a root-only audit easily misses) still references it
via ``FeatureFlag.<NAME>`` — e.g. ``USE_GRAPH_RAG`` / ``USE_MULTI_HOP_RETRIEVAL``
are read by ``verticals/victor-coding/.../code_search_tool.py``. Removing such a
flag makes the reference raise ``AttributeError`` at runtime (a latent breakage
unit tests don't catch unless they exercise that path).

Scans ``victor/``, ``verticals/``, and ``victor-contracts/`` for
``FeatureFlag.<NAME>`` references and asserts each NAME is a real enum member.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from victor.core.feature_flags import FeatureFlag

# FeatureFlag.<NAME> — upper-snake identifier after the dot.
_FLAG_REF = re.compile(r"FeatureFlag\.([A-Z][A-Z0-9_]*)")

# Scan roots + file suffixes. Exclude tests (they legitimately reference flags)
# and the enum/defaults definitions themselves.
_SCAN_ROOTS = ("victor", "verticals", "victor-contracts")
_EXCLUDE_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "test",
}
_EXCLUDE_FILES = {"feature_flags.py", "feature_config.py"}


def _iter_source_files(repo_root: Path):
    for root_name in _SCAN_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _EXCLUDE_DIRS for part in path.parts):
                continue
            if path.name in _EXCLUDE_FILES:
                continue
            yield path


def _collect_refs(repo_root: Path) -> dict[str, list[str]]:
    """Return {referenced_name: [file:line, ...]} for every FeatureFlag.<NAME>."""
    refs: dict[str, list[str]] = {}
    for path in _iter_source_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comments / docstrings heuristically (cheap + good enough).
            stripped = line.lstrip()
            if stripped.startswith("#") or '"""' in line or "'''" in line:
                continue
            for name in _FLAG_REF.findall(line):
                refs.setdefault(name, []).append(f"{path}:{lineno}")
    return refs


_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_every_referenced_feature_flag_exists():
    """No production code may reference a FeatureFlag that isn't defined.

    Catches the "removed a flag a vertical still uses" regression (e.g. the
    USE_GRAPH_RAG removal in #539 broke victor-coding's code_search_tool).
    """
    valid = set(FeatureFlag.__members__)
    refs = _collect_refs(_REPO_ROOT)
    missing = {name: locs for name, locs in refs.items() if name not in valid}
    if missing:
        detail = "\n".join(
            f"  FeatureFlag.{name} (referenced at {locs[0]}, +{len(locs) - 1} more)"
            for name, locs in sorted(missing.items())
        )
        pytest.fail(
            "Production code references FeatureFlag members that are not in the "
            f"FeatureFlag enum:\n{detail}\nEither restore the flag(s) to the enum "
            "(victor/core/feature_flags.py) or update the references."
        )


def test_graph_flags_referenced_by_victor_coding_exist():
    """Pin the specific regression: the graph flags victor-coding reads must exist.

    USE_GRAPH_RAG + USE_MULTI_HOP_RETRIEVAL are read by
    verticals/victor-coding/victor_coding/tools/code_search_tool.py; removing
    them caused a latent AttributeError. This explicit assertion survives even
    if the general scan above is bypassed.
    """
    assert hasattr(FeatureFlag, "USE_GRAPH_RAG")
    assert hasattr(FeatureFlag, "USE_MULTI_HOP_RETRIEVAL")
