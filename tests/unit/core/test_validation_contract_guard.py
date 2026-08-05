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

"""FEP-0014 guard: no NEW duplicate ValidationSeverity / ValidationResult.

The canonical ``ValidationSeverity`` and ``ValidationResult`` live in
``victor/core/validation.py``. This guard scans every ``victor/**/*.py`` module for
class definitions with those names and fails if one appears anywhere other than the
canonical module OR the explicit ALLOWLIST of already-known duplicates.

The allowlist captures the divergent definitions that FEP-0014 Phase 2 will migrate
onto the canonical types. It **shrinks toward empty** as each consumer is migrated;
when Phase 3 removes the last shim the allowlist is empty and this guard enforces a
single source of truth going forward. The test PASSES today (allowlist covers the
known duplicates) and FAILS the moment a *new* duplicate is introduced.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Repository-relative path to the victor package (…/victor).
_VICTOR_PKG = Path(__file__).resolve().parents[3] / "victor"

# The one canonical home for both symbols; never counts as a duplicate.
_CANONICAL = "victor/core/validation.py"

# Known pre-existing duplicate locations, as POSIX-style paths relative to the repo
# root. FEP-0014 Phase 2 migrates each of these onto the canonical types, removing it
# from this allowlist; the set is expected to reach {} by the end of Phase 3.
_ALLOWLIST: dict[str, set[str]] = {
    # FEP-0014 Phase 2a (done): all three ValidationSeverity duplicates now
    # re-export the canonical enum, so the allowlist is empty — the guard still
    # scans for (and rejects) any NEW ValidationSeverity duplicate.
    "ValidationSeverity": set(),
    # FEP-0014 Phase 2b/2c (done): the four divergent ValidationResult types were
    # distinct concepts sharing a name, so each was renamed to a domain-specific
    # name (ToolCallValidationResult, ConnectionValidationResult, RequirementResult,
    # CapabilityValidationResult) rather than merged into the canonical type. No
    # ValidationResult duplicates remain outside canonical — the guard still scans
    # for (and rejects) any NEW ValidationResult duplicate.
    "ValidationResult": set(),
}

_TARGET_NAMES = frozenset(_ALLOWLIST)


def _relpath(path: Path) -> str:
    """Return a POSIX repo-relative path like ``victor/core/validation.py``."""
    return path.relative_to(_VICTOR_PKG.parent).as_posix()


def _find_class_defs() -> dict[str, set[str]]:
    """Map each target class name to the set of modules that define it.

    Walks the full AST of every module so nested class definitions (e.g. a class
    declared inside a function or another class) are also detected.
    """
    found: dict[str, set[str]] = {name: set() for name in _TARGET_NAMES}
    for py_file in _VICTOR_PKG.rglob("*.py"):
        rel = _relpath(py_file)
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in _TARGET_NAMES:
                found[node.name].add(rel)
    return found


def test_no_new_duplicate_validation_classes() -> None:
    """No ValidationSeverity/ValidationResult outside canonical + allowlist."""
    found = _find_class_defs()
    offenders: dict[str, set[str]] = {}
    for name, locations in found.items():
        allowed = {_CANONICAL} | _ALLOWLIST[name]
        unexpected = locations - allowed
        if unexpected:
            offenders[name] = unexpected

    assert not offenders, (
        "New duplicate validation class(es) introduced outside "
        f"{_CANONICAL} and the FEP-0014 allowlist: {offenders}. "
        "Import the canonical type from victor.core.validation instead."
    )


def test_canonical_module_defines_both_symbols() -> None:
    """The canonical module actually defines both symbols (allowlist sanity)."""
    found = _find_class_defs()
    assert _CANONICAL in found["ValidationSeverity"]
    assert _CANONICAL in found["ValidationResult"]


def test_allowlist_entries_still_exist() -> None:
    """Every allowlisted duplicate is still present.

    Guards against a stale allowlist: once Phase 2 removes a duplicate, its entry
    should be deleted from ``_ALLOWLIST`` too, keeping the list honestly shrinking.
    """
    found = _find_class_defs()
    for name, allowed in _ALLOWLIST.items():
        stale = allowed - found[name]
        assert not stale, (
            f"Allowlist for {name} lists modules that no longer define it: {stale}. "
            "Remove them from _ALLOWLIST (FEP-0014 Phase 2 migration progress)."
        )
