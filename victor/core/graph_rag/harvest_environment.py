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

"""Constrain the python tier-3 harvest to what the codebase imports.

Measured motivation (2026-10-25, recorded in native-acceleration-strategy):
the scip-python harvest ran 2,980 s — half the full-index wall clock —
because its embedded Pyright resolves imports against the RUNNING
interpreter's environment: on a shared conda env that means walking ~17 GB
of site-packages (with WAL writeback thrashing the page cache, ~130 GB of
real disk reads).

``scip-python index --environment <json>`` replaces its whole-environment
enumeration with a caller-provided package list (schema verified against
the installed 0.6.x bundle: ``[{"name", "version", "files"}, ...]`` — the
same shape its own ``get_packages.py`` produces). This module derives that
list from the CODEBASE: an AST import scan (the same tree-parse basis the
earlier passes use) over the repo, stdlib and repo-local roots removed,
mapped to installed distributions plus their dependency closure. The
harvest then resolves third-party imports against exactly that surface.

Reflection is part of the contract: the builder reports how much of the
environment was selected vs installed and which import roots stayed
unmapped, so a constrained run's join rates can be compared against the
unconstrained baseline honestly. Decline, never break: ANY failure returns
None and the harvest runs unconstrained.
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git",
    ".victor",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "target",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def collect_import_roots(root: Path) -> set[str]:
    """Top-level imported module names across the repo's python files.

    The literal "tree parse based imports" of the earlier passes: one
    stdlib-ast walk per file (C parse), ``Import`` names and non-relative
    ``ImportFrom`` module roots only — the resolution universe Pyright is
    about to be asked for.
    """
    roots: set[str] = set()
    if not root.is_dir():
        return roots
    for path in sorted(root.rglob("*.py")):
        if _SKIP_DIRS.intersection(path.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    roots.add(node.module.split(".")[0])
    return roots


def repo_local_roots(root: Path) -> set[str]:
    """Import roots the repo itself provides.

    Top-level packages plus the monorepo's nested first-party packages
    (``<dir>-<name>/<dir>_<name>/__init__.py`` at depth two —
    victor-codegraph/victor_codegraph, verticals/victor-coding/
    victor_coding, ...): those import roots are the repo, not the
    environment.
    """
    roots: set[str] = set()
    if not root.is_dir():
        return roots
    for entry in root.iterdir():
        if entry.is_dir() and (entry / "__init__.py").is_file():
            roots.add(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            roots.add(entry.stem)
    for entry in root.iterdir():
        if not entry.is_dir() or "-" not in entry.name:
            continue
        pkg = entry / entry.name.replace("-", "_")
        if (pkg / "__init__.py").is_file():
            roots.add(pkg.name)
        elif entry.name.replace("-", "_") == entry.name and (entry / "__init__.py").is_file():
            roots.add(entry.name)
    for parent in (root / "verticals",):
        if parent.is_dir():
            for entry in parent.iterdir():
                pkg = entry / entry.name.replace("-", "_")
                if entry.is_dir() and (pkg / "__init__.py").is_file():
                    roots.add(pkg.name)
    return roots


def _requires_names(distribution: Any) -> list[str]:
    """Distribution names from ``Requires-Dist``, markers/extras stripped."""
    requires = distribution.requires or []
    names = []
    for entry in requires:
        name = entry.split(";")[0].split("[")[0].strip()
        if name:
            names.append(name)
    return names


def select_distributions(
    import_roots: set[str],
    *,
    packages_distributions: Optional[dict[str, list[str]]] = None,
    distribution_reader: Optional[Callable[[str], Any]] = None,
) -> tuple[set[str], list[str]]:
    """Distributions covering ``import_roots`` plus their dependency closure.

    Returns (selected distribution names, unmapped import roots). Pure
    function of its inputs — the metadata readers are injectable so tests
    are deterministic and never touch the real environment.
    """
    import importlib.metadata as metadata

    if packages_distributions is None:
        packages_distributions = metadata.packages_distributions()
    if distribution_reader is None:
        distribution_reader = metadata.distribution

    unmapped: list[str] = []
    selected: dict[str, str] = {}  # canonical name -> provenance ("import:root" | "dep:name")
    frontier: list[str] = []
    for root_name in sorted(import_roots):
        dists = packages_distributions.get(root_name)
        if not dists:
            unmapped.append(root_name)
            continue
        for dist_name in dists:
            canonical = _normalize(dist_name)
            if canonical not in selected:
                selected[canonical] = f"import:{root_name}"
                frontier.append(dist_name)

    while frontier:
        dist_name = frontier.pop()
        try:
            distribution = distribution_reader(dist_name)
        except Exception:  # unresolvable requirement name — skip, never break
            continue
        for required in _requires_names(distribution):
            canonical = _normalize(required)
            if canonical in selected:
                continue
            try:
                distribution_reader(canonical)  # installed?
            except Exception:
                continue  # optional/absent dependency
            selected[canonical] = f"dep:{_normalize(dist_name)}"
            frontier.append(required)

    return set(selected), unmapped, selected


def _normalize(name: str) -> str:
    return name.strip().replace("_", "-").lower()


def write_environment_json(
    path: Path,
    distributions: set[str],
    *,
    distribution_reader: Optional[Callable[[str], Any]] = None,
) -> Optional[Path]:
    """Emit the scip-python environment file; None (decline) when invalid.

    The schema is exactly what scip-python's own enumeration produces:
    ``[{"name": str, "version": str, "files": [str, ...]}, ...]``.
    """
    import importlib.metadata as metadata

    if distribution_reader is None:
        distribution_reader = metadata.distribution

    payload: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for dist_name in sorted(distributions, key=str.lower):
        try:
            distribution = distribution_reader(dist_name)
            files = [str(f) for f in (distribution.files or [])]
            display = distribution.metadata["Name"] or dist_name
            if _normalize(display) in seen_names:
                continue  # case-variant duplicate of an already-emitted dist
            seen_names.add(_normalize(display))
            payload.append({"name": display, "version": distribution.version, "files": files})
        except Exception:
            logger.debug("environment constraint: skipping unresolvable %s", dist_name)

    if not payload:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        # Round-trip validation: a malformed file would crash the harvest
        # child (DumpSkipped) — worse than running unconstrained.
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not (
            isinstance(parsed, list)
            and parsed
            and all(
                isinstance(item, dict)
                and {"name", "version", "files"} <= set(item)
                and isinstance(item["files"], list)
                for item in parsed
            )
        ):
            raise ValueError("environment payload failed schema validation")
    except (OSError, ValueError) as exc:
        logger.warning("environment constraint declined: %s", exc)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return path


def build_python_environment(root: Path, output_path: Path) -> Optional[Path]:
    """The full constraint pipeline; None means harvest unconstrained."""
    try:
        import_roots = collect_import_roots(root)
        if not import_roots:
            return None
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        local = repo_local_roots(root)
        third_party = import_roots - stdlib - local
        if not third_party:
            return None
        selected, unmapped, provenance = select_distributions(third_party)
        if not selected:
            return None
        environment_path = write_environment_json(output_path, selected)
        if environment_path is None:
            return None
        import importlib.metadata as metadata

        installed = sum(1 for _ in metadata.distributions())
        file_count = sum(1 for _ in open(environment_path, encoding="utf-8").read().splitlines())
        direct = sorted(n for n, why in provenance.items() if why.startswith("import:"))
        by_dependency = len(selected) - len(direct)
        logger.info(
            "python harvest constrained: %d import roots -> %d distributions "
            "(%d direct + %d dependency closure, of %d installed; env file "
            "~%d lines); unmapped roots: %s",
            len(third_party),
            len(selected),
            len(direct),
            by_dependency,
            installed,
            file_count,
            ", ".join(sorted(unmapped)[:10]) or "none",
        )
        return environment_path
    except Exception as exc:  # decline, never break the harvest
        logger.warning("environment constraint failed (%s); harvesting unconstrained", exc)
        return None


def external_packages_from_index(index: Any) -> set[str]:
    """Environment packages a SCIP dump actually resolved.

    scip-python emits environment-package symbols as
    ``scip-python python <package> <version> `module`...`` — the compiled
    ground truth of what Pyright pulled in. This is the reflection basis:
    the constrained run must cover (at least) what the baseline resolved.
    """
    packages: set[str] = set()
    prefix = "scip-python python "
    # SCIP keeps environment-package symbols on Index.external_symbols
    # (empirically 1,386 entries on the victor baseline dump), NOT on a
    # symbols field — reading the right member is what makes this
    # reflection observably non-empty.
    symbols = getattr(index, "external_symbols", None) or getattr(index, "symbols", ()) or ()
    for symbol in symbols:
        name = getattr(symbol, "symbol", "")
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        parts = rest.split(" ", 2)
        if len(parts) >= 2 and parts[0] and parts[1]:
            packages.add(_normalize(parts[0]))
    return packages


def reflect_harvest(
    baseline_index: Any,
    constrained_index: Any,
    selected_distributions: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Compare a constrained dump against its unconstrained baseline.

    The dynamic half of the reflection contract: coverage is judged by the
    dumps' own resolved-package sets, not by declared intent. A package the
    baseline resolved that the constrained run lost is a real regression to
    warn (and potentially re-run) on — never a silent degradation.
    """
    baseline = external_packages_from_index(baseline_index)
    constrained = external_packages_from_index(constrained_index)
    missing = baseline - constrained
    added = constrained - baseline if baseline else set()
    reflection: dict[str, Any] = {
        "baseline_packages": len(baseline),
        "constrained_packages": len(constrained),
        "missing_vs_baseline": sorted(missing),
        "newly_resolved": sorted(added),
        "baseline_occurrences": sum(
            len(getattr(doc, "occurrences", ()) or ()) for doc in baseline_index.documents
        ),
        "constrained_occurrences": sum(
            len(getattr(doc, "occurrences", ()) or ()) for doc in constrained_index.documents
        ),
    }
    if selected_distributions is not None:
        reflection["selected_distributions"] = len(selected_distributions)
    return reflection
