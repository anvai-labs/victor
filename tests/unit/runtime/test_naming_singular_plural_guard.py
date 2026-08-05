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

"""Guard test for directory naming consistency (singular vs plural packages).

Enforces the repository's package-naming convention by rejecting *new*
same-parent singular/plural directory collisions — e.g. a parent that contains
both ``foo/`` and ``foos/`` for the same concept. This is the exact drift that
once produced ``tests/benchmark/`` alongside ``tests/benchmarks/`` and
``victor/integrations/protocol/`` alongside ``victor/integrations/protocols/``.

Naming convention (authoritative statement — this test is the source of truth):
  1. Top-level packages and their ``tests/`` mirrors are **plural** when they
     hold a collection of modules (``victor/tools``, ``victor/providers``,
     ``victor/protocols``, ``victor/verticals``, ``tests/unit/tools`` ...).
     A test directory mirrors its source package name *exactly*.
  2. **Singular** is reserved for a sub-package that describes one concrete
     subsystem or whose name is a proper/mass noun
     (``victor/benchmark`` the vertical, ``mode_config``, ``profiler``).
  3. A *category* test directory that is NOT a package mirror uses a descriptive
     name, not the bare noun (``tests/performance``, not ``tests/benchmark``).
  4. A parent directory MUST NOT contain both the singular and the plural form
     of the same concept. If two distinct concepts share a stem, rename one so
     the intent is unambiguous (e.g. ``agents/`` meaning multi-agent -> ``multi_agent/``).

This guard only flags same-parent pairs. Cross-layer splits where singular and
plural describe genuinely different things at different layers
(e.g. ``victor/agent/conversation`` the runtime store vs
``victor/framework/conversations`` the coordination framework) are NOT same-parent
and are therefore allowed; the distinction is documented, not mechanical.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
"""Absolute path to the repository root (tests/unit/runtime/<this file>)."""

# Trees whose package layout this guard enforces.
SCOPED_ROOTS = [REPO_ROOT / "victor", REPO_ROOT / "tests"]

# Directory names to ignore while walking (caches, build artifacts, venvs).
SKIP_DIRS = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    "node_modules",
    "site",
    "build",
    "dist",
    "htmlcov",
    ".venv",
    "embeddings.lance",
    ".victor",
}

# Grandfathered same-parent singular/plural pairs that are allowed to remain.
# Each entry is (parent_path_relative_to_repo, singular_name, plural_name).
#
# The codebase currently has NONE — keep it that way. Add an entry only if a pair
# represents two genuinely distinct concepts that cannot be disambiguated by
# renaming, and leave a comment explaining why. Every resolution should REMOVE an
# entry rather than add one, mirroring how the singleton cap is meant to shrink.
ALLOWED_SAME_PARENT_PAIRS: set[tuple[str, str, str]] = set()


def _plural_candidates(name: str) -> list[str]:
    """Return plausible plural spellings of ``name`` (English heuristics)."""
    candidates: list[str] = []
    if len(name) > 1 and name.endswith("y") and name[-2] not in "aeiou":
        candidates.append(name[:-1] + "ies")
    if name.endswith(("s", "x", "z", "ch", "sh")):
        candidates.append(name + "es")
    candidates.append(name + "s")
    return candidates


def _find_same_parent_singular_plural_pairs() -> list[tuple[str, str, str]]:
    """Return ``(parent_rel, singular, plural)`` for every same-parent collision."""
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for root in SCOPED_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            present = set(dirnames)
            parent_rel = str(Path(dirpath).relative_to(REPO_ROOT))
            for name in dirnames:
                for plural in _plural_candidates(name):
                    if plural != name and plural in present:
                        key = (parent_rel, name, plural)
                        if key not in seen:
                            seen.add(key)
                            pairs.append(key)
    return sorted(pairs)


class TestNamingSingularPluralGuard:
    """Prevent new same-parent singular/plural directory collisions."""

    def test_no_new_same_parent_singular_plural_dirs(self) -> None:
        """Fail if a parent dir holds both ``foo`` and ``foos``.

        Same-parent singular/plural pairs are a naming smell: two directories
        that look like variants of one concept. Rename one so the intent is
        unambiguous (see the module docstring for the convention), or, if the
        pair is genuinely two distinct concepts, document it by adding the
        ``(parent, singular, plural)`` triple to ``ALLOWED_SAME_PARENT_PAIRS``.
        """
        violations = [
            (parent, singular, plural)
            for (parent, singular, plural) in _find_same_parent_singular_plural_pairs()
            if (parent, singular, plural) not in ALLOWED_SAME_PARENT_PAIRS
        ]
        if not violations:
            return
        detail = "\n".join(
            f"  {parent}/{singular}  <->  {parent}/{plural}"
            for parent, singular, plural in violations
        )
        pytest.fail(
            f"Found {len(violations)} same-parent singular/plural directory pair(s):\n{detail}\n\n"
            "A parent must not contain both the singular and plural form of one concept. "
            "Rename one for clarity (see this test's module docstring for the convention), "
            "or add the triple to ALLOWED_SAME_PARENT_PAIRS with a justifying comment."
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        """Fail if an allowlisted pair no longer exists on disk.

        Keeps the allowlist honest: when a grandfathered collision is resolved,
        its entry must be removed rather than left to rot.
        """
        live = set(_find_same_parent_singular_plural_pairs())
        stale = sorted(ALLOWED_SAME_PARENT_PAIRS - live)
        if not stale:
            return
        detail = "\n".join(
            f"  {parent}/{singular} <-> {parent}/{plural}" for parent, singular, plural in stale
        )
        pytest.fail(
            f"ALLOWED_SAME_PARENT_PAIRS has {len(stale)} stale entr(y/ies) — the pair(s) no "
            f"longer exist on disk:\n{detail}\nRemove them; the allowlist must only contain "
            "currently-present pairs."
        )
