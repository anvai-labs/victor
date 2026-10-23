#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Layering ratchet: upward imports onto concrete implementations may only shrink.

Victor's layers (docs/architecture.md): Client -> Framework -> Runtime ->
Infrastructure, each depending only on the layers below. Nothing enforced
the runtime->framework or infrastructure->runtime directions, so concrete
upward dependencies accumulated unseen (victor's own tier-3 code graph
measured 460 such edges).

This scans every import statement under ``victor/`` (module-level AND
function-local; ``if TYPE_CHECKING:`` blocks are skipped) and reports each
upward import whose imported names are not all abstractions. Importing a
Protocol, ABC, Enum, TypedDict, NamedTuple or dataclass from a higher layer
is Dependency Inversion done right (the higher layer owns the contract) and
is allowed. So are the design-sanctioned edges listed in ``SANCTIONED``.

Known blind spots (by design — static AST only): string-based imports
(``importlib.import_module``), lazy ``__getattr__`` re-export maps, and
attribute access through an already-imported package are not seen; and the
baseline is per (importer, imported) MODULE pair, so a new name imported
through an already-baselined pair is not flagged. Exception classes count as
concrete: a lower layer catching a higher layer's exception by class is
coupling, not inversion.

The committed baseline (``scripts/ci/layering_baseline.json``) holds the
known (importer module, imported module) pairs:

- a pair not in the baseline -> FAIL (a new upward leak);
- a baseline pair that no longer occurs -> FAIL until the baseline is
  regenerated with ``--update-baseline`` (the improvement is locked in, the
  same contract as the duplication ratchet).

Usage::

    python scripts/ci/layering_ratchet.py                  # check
    python scripts/ci/layering_ratchet.py --update-baseline
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "victor"
BASELINE_PATH = ROOT / "scripts" / "ci" / "layering_baseline.json"

# Canonical layer map, longest prefix wins. Packages not listed (analytics,
# benchmark, evaluation, observability, verticals, ...) are cross-cutting or
# consumer packages and are not ratcheted.
LAYER_PREFIXES: dict[str, str] = {
    "victor.ui": "client",
    "victor.integrations": "client",
    # The MCP client/registry is a protocol client consumed by the runtime
    # and tools (infrastructure, like a provider adapter); only the server
    # (victor exposed AS an MCP server) is a client-layer surface.
    "victor.integrations.mcp": "infrastructure",
    "victor.integrations.mcp.server": "client",
    "victor.framework": "framework",
    "victor.agent": "runtime",
    "victor.teams": "runtime",
    "victor.workflows": "runtime",
    "victor.runtime": "runtime",
    "victor.context": "runtime",
    "victor.providers": "infrastructure",
    "victor.tools": "infrastructure",
    "victor.state": "infrastructure",
    "victor.storage": "infrastructure",
    "victor.core": "infrastructure",
    "victor.config": "infrastructure",
    "victor.processing": "infrastructure",
    "victor.native": "infrastructure",
    "victor.security": "infrastructure",
    "victor.protocols": "infrastructure",
    "victor.models": "infrastructure",
    "victor.ml": "infrastructure",
    "victor.classification": "infrastructure",
}
RANK = {"client": 0, "framework": 1, "runtime": 2, "infrastructure": 3}

# Design-sanctioned upward edges: (importer prefix, imported prefix). An
# imported prefix ending in "." matches that package and its submodules only.
SANCTIONED: tuple[tuple[str, str], ...] = (
    # Teams are formations used directly as StateGraph nodes, and YAML
    # workflows compile to StateGraph (CLAUDE.md architecture facts).
    ("victor.teams", "victor.framework.graph"),
    ("victor.workflows", "victor.framework.graph"),
    # The RL engine (coordinator, learners, credit assignment) is a shared
    # engine the runtime drives, not public API layered above it. Lower
    # layers still reach it only through victor.core.rl.coordinator_port.
    *(
        (runtime, "victor.framework.rl.")
        for runtime in (
            "victor.agent",
            "victor.teams",
            "victor.workflows",
            "victor.runtime",
            "victor.context",
        )
    ),
    # Composition roots wire concrete services from every layer by design.
    ("victor.core.bootstrap", ""),
    ("victor.core.bootstrap_services", ""),
)

_ABSTRACT_BASES = {
    "Protocol",
    "ABC",
    "Enum",
    "IntEnum",
    "StrEnum",
    "Flag",
    "IntFlag",
    "TypedDict",
    "NamedTuple",
}


def layer_of(module: str) -> Optional[str]:
    """Layer of a dotted module name (longest matching prefix), or None."""
    best: Optional[str] = None
    best_len = -1
    for prefix, layer in LAYER_PREFIXES.items():
        if (module == prefix or module.startswith(prefix + ".")) and len(prefix) > best_len:
            best, best_len = layer, len(prefix)
    return best


def _is_sanctioned(importer: str, imported: str) -> bool:
    for src, dst in SANCTIONED:
        if (importer == src or importer.startswith(src + ".")) and (
            not dst or imported == dst.rstrip(".") or imported.startswith(dst)
        ):
            return True
    return False


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


@lru_cache(maxsize=None)
def _module_file(module: str) -> Optional[Path]:
    base = ROOT.joinpath(*module.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


@lru_cache(maxsize=None)
def _abstract_names(module: str) -> frozenset[str]:
    """Top-level classes in ``module`` that are abstractions (DIP contracts)."""
    path = _module_file(module)
    if path is None:
        return frozenset()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return frozenset()
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {
            b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else ""
            for b in node.bases
        }
        decos = {
            (
                d.id
                if isinstance(d, ast.Name)
                else (
                    d.attr
                    if isinstance(d, ast.Attribute)
                    else (
                        (d.func.id if isinstance(d.func, ast.Name) else getattr(d.func, "attr", ""))
                        if isinstance(d, ast.Call)
                        else ""
                    )
                )
            )
            for d in node.decorator_list
        }
        if bases & _ABSTRACT_BASES or decos & {"runtime_checkable", "dataclass"}:
            names.add(node.name)
    return frozenset(names)


def _type_checking_lines(tree: ast.AST) -> set[int]:
    """Line numbers of statements inside ``if TYPE_CHECKING:`` blocks."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            name = test.id if isinstance(test, ast.Name) else getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                for child in node.body:
                    for sub in ast.walk(child):
                        if hasattr(sub, "lineno"):
                            lines.add(sub.lineno)
    return lines


def _resolve(importer: str, is_pkg: bool, node: ast.ImportFrom) -> Optional[str]:
    if node.level == 0:
        return node.module
    parts = importer.split(".")
    if not is_pkg:
        parts = parts[:-1]
    up = node.level - 1
    if up > len(parts):
        return None
    base = parts[: len(parts) - up] if up else parts
    return ".".join(base + ([node.module] if node.module else []))


def iter_upward_leaks(files: Optional[Iterable[Path]] = None) -> Iterator[tuple[str, str]]:
    """Yield (importer module, imported module) for every concrete upward import."""
    for path in sorted(files if files is not None else PKG.rglob("*.py")):
        importer = _module_name(path)
        src_layer = layer_of(importer)
        if src_layer is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        skip = _type_checking_lines(tree)
        is_pkg = path.name == "__init__.py"
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) in skip:
                continue
            targets: list[tuple[str, list[str]]] = []
            if isinstance(node, ast.Import):
                targets = [(a.name, []) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = _resolve(importer, is_pkg, node)
                if mod:
                    targets = [(mod, [a.name for a in node.names])]
            for imported, names in targets:
                dst_layer = layer_of(imported)
                if dst_layer is None or RANK[src_layer] <= RANK[dst_layer]:
                    continue
                if _is_sanctioned(importer, imported):
                    continue
                # A name that is itself a submodule is a concrete module import.
                abstract = _abstract_names(imported)
                if names and all(n in abstract for n in names):
                    continue
                yield importer, imported


def current_pairs() -> set[str]:
    return {f"{a} -> {b}" for a, b in iter_upward_leaks()}


def load_baseline(path: Path = BASELINE_PATH) -> set[str]:
    try:
        return set(json.loads(path.read_text(encoding="utf-8"))["pairs"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"layering-ratchet: baseline {path} missing or malformed ({exc}); "
            "regenerate with `python scripts/ci/layering_ratchet.py --update-baseline`"
        ) from exc


def write_baseline(pairs: set[str], path: Path = BASELINE_PATH) -> None:
    payload = {"count": len(pairs), "pairs": sorted(pairs)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def check(current: set[str], baseline: set[str]) -> int:
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)
    print(
        f"layering-ratchet: {len(current)} upward concrete import pairs "
        f"(baseline {len(baseline)}; {len(new)} new, {len(fixed)} fixed)"
    )
    for pair in new:
        print(f"  NEW upward leak: {pair}")
    if new:
        print(
            "FAIL: a lower layer imports a concrete implementation from a higher "
            "layer. Move the shared code down, depend on a Protocol owned by the "
            "higher layer, or inject the dependency (docs/architecture.md)."
        )
        return 1
    if fixed:
        for pair in fixed:
            print(f"  fixed: {pair}")
        print(
            "FAIL: upward leaks were removed — lock the improvement in with "
            "`python scripts/ci/layering_ratchet.py --update-baseline`."
        )
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)
    pairs = current_pairs()
    if args.update_baseline:
        write_baseline(pairs, BASELINE_PATH)
        print(f"baseline written: {len(pairs)} pairs -> {BASELINE_PATH}")
        return 0
    return check(pairs, load_baseline(BASELINE_PATH))


if __name__ == "__main__":
    sys.exit(main())
