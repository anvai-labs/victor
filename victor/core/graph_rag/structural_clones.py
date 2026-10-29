# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Structural near-duplicate discovery: logic that is almost identical.

``semantic_clones`` embeds a symbol's name, signature and docstring, so it
pairs a Protocol with its implementation and misses copies whose names
drifted. This module compares function *bodies* instead:

- Each body becomes a token stream (pre-order AST walk, docstring dropped).
  Locals and parameters are renamed to placeholders in first-use order,
  constants collapse to their type, and every other name keeps its
  **resolved SCIP symbol**, so ``registry.get`` and ``cache.get`` stay
  distinct even though both are spelled ``.get``.
- ``exact``: identical streams in different files (copies with renamed
  variables or changed constants).
- ``template``: identical once resolved callees are erased too, while the
  callees differ (a kernel with an injected callable).
- ``near``: MinHash/LSH Jaccard over 5-token shingles at or above a
  threshold (copies with a few statements added, removed or edited).

Families whose members are all sanctioned (Protocol or abstract stubs,
literal-dense data tables) are reported as suppressed with the reason,
never dropped silently. Documents whose SCIP occurrences no longer match
the source fall back to AST-only resolution and are counted as stale.
Read-only analysis; surfaced as ``victor graph similar --mode structural``.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from victor.core.graph_rag.semantic_clones import (
    MAX_PAIRS,
    TEST_PATH_MARKERS,
    TooManyPairsError,
    make_union_find,
)

DEFAULT_MIN_NODES = 40
DEFAULT_NEAR_THRESHOLD = 0.8
LITERAL_DENSE_RATIO = 0.4
SHINGLE = 5
NUM_PERM = 128
LSH_ROWS = 4
SCIP_PYTHON_DUMP = Path(".victor") / "precise" / "scip-python.scip"
_IDENT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")  # dotted: import paths
_PRIME = 4_294_967_311  # smallest prime above 2**32


@dataclass(frozen=True)
class FunctionUnit:
    file_path: str
    name: str
    line: int
    size: int
    tokens: Tuple[str, ...]
    exact_hash: str
    shape_hash: str
    sanctioned: Optional[str] = None


@dataclass
class StructuralFamily:
    tier: str  # exact | template | near
    score: float
    members: List[FunctionUnit] = field(default_factory=list)
    suppressed: Optional[str] = None

    @property
    def files(self) -> List[str]:
        return sorted({m.file_path for m in self.members})


@dataclass
class StructuralReport:
    families: List[StructuralFamily]
    suppressed: List[StructuralFamily]
    stats: Dict[str, Any]


# -- SCIP occurrence join ----------------------------------------------------


OccurrenceMap = Dict[Tuple[int, int], Tuple[int, str]]  # (line1, col) -> (length, symbol)


def load_scip_occurrences(index_path: Path) -> Optional[Dict[str, List[Tuple[int, int, int, str]]]]:
    """``relative_path -> [(line0, col, end_col, symbol)]`` for single-line ranges.

    ``None`` when the dump is absent or unreadable (callers fall back to
    AST-only resolution).
    """
    if not index_path.is_file():
        return None
    from victor.core.graph_rag.precise_index import _read_index

    index = _read_index(index_path)
    if index is None:
        return None
    out: Dict[str, List[Tuple[int, int, int, str]]] = {}
    for doc in index.documents:
        occs = []
        for occ in doc.occurrences:
            rng = list(occ.range)
            if len(rng) == 3:  # [line, start_col, end_col]: single-line identifier
                occs.append((rng[0], rng[1], rng[2], occ.symbol))
        out[doc.relative_path] = occs
    return out


def fresh_occurrence_map(
    lines: Sequence[str], occurrences: Sequence[Tuple[int, int, int, str]]
) -> Optional[OccurrenceMap]:
    """Map ``(line1, col) -> (length, symbol)`` if the ranges still match the source.

    A dump built from an older revision points most of its ranges at other
    text; more than 20% (and over 5) non-identifier ranges marks the
    document stale. Below that, mismatches are SCIP quirks (f-string
    offsets, ``import x as y`` spans) that never coincide with a body name,
    and :func:`normalize_function` re-checks every lookup's length anyway.
    """
    mapping: OccurrenceMap = {}
    bad = 0
    for line0, col, end, symbol in occurrences:
        text = lines[line0][col:end] if line0 < len(lines) else ""
        if not _IDENT.fullmatch(text or "_"):
            bad += 1
        mapping[(line0 + 1, col)] = (end - col, symbol)
    if bad > max(5, int(len(occurrences) * 0.2)):
        return None
    return mapping


def _descriptor(symbol: str) -> str:
    """Drop the ``<scheme> <manager> <package> <version>`` prefix."""
    parts = symbol.split(" ", 4)
    return parts[4] if len(parts) == 5 else symbol


# -- normalisation -----------------------------------------------------------


def _body(fn: ast.AST) -> List[ast.stmt]:
    body = list(getattr(fn, "body", []))
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


def _preorder(node: ast.AST) -> Iterator[ast.AST]:
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _preorder(child)


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def normalize_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, occurrences: Optional[OccurrenceMap]
) -> List[str]:
    """Token stream for ``fn``'s body (see the module docstring)."""
    module = ast.Module(body=_body(fn), type_ignores=[])
    stored = {
        n.id for n in ast.walk(module) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    local_names = _param_names(fn) | stored
    own_scope = f"{fn.name}()."
    placeholders: Dict[str, str] = {}

    def placeholder(key: str) -> str:
        if key not in placeholders:
            placeholders[key] = f"v{len(placeholders)}"
        return placeholders[key]

    def resolve(key: Tuple[int, int], name: str) -> Optional[str]:
        # only a range spanning exactly this name: a stale or shifted dump misses
        hit = occurrences.get(key) if occurrences else None
        return hit[1] if hit is not None and hit[0] == len(name) else None

    tokens: List[str] = []
    for node in _preorder(module):
        if node is module:
            continue
        if isinstance(node, ast.Name):
            symbol = resolve((node.lineno, node.col_offset), node.id)
            if symbol is not None:
                if symbol.startswith("local ") or own_scope in symbol:
                    tokens.append(placeholder(symbol))
                else:
                    tokens.append("G:" + _descriptor(symbol))
            elif node.id in local_names:
                tokens.append(placeholder("name:" + node.id))
            else:
                tokens.append("G:" + node.id)
        elif isinstance(node, ast.Attribute):
            symbol = None
            if node.end_lineno is not None and node.end_col_offset is not None:
                symbol = resolve((node.end_lineno, node.end_col_offset - len(node.attr)), node.attr)
            target = _descriptor(symbol) if symbol else node.attr
            if isinstance(node.value, ast.Name) and node.value.id == "self" and "#" in target:
                target = "Self#" + target.split("#", 1)[1]  # the copy lives in another class
            tokens.append("A:" + target)
        elif isinstance(node, ast.Constant):
            tokens.append("C:" + type(node.value).__name__)
        elif isinstance(node, ast.arg):
            tokens.append(placeholder("name:" + node.arg))
        else:
            tokens.append(type(node).__name__)
    return tokens


def _shape(tokens: Sequence[str]) -> List[str]:
    """Erase resolved globals and attributes: the callee-agnostic shape."""
    return ["G" if t.startswith("G:") else "A" if t.startswith("A:") else t for t in tokens]


def _digest(tokens: Sequence[str]) -> str:
    return hashlib.blake2b("\x1f".join(tokens).encode("utf-8"), digest_size=12).hexdigest()


def _sanction(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, enclosing: Optional[ast.ClassDef], tokens: List[str]
) -> Optional[str]:
    if enclosing is not None and any(
        (isinstance(b, ast.Name) and b.id == "Protocol")
        or (isinstance(b, ast.Attribute) and b.attr == "Protocol")
        for b in enclosing.bases
    ):
        return "protocol"
    for dec in fn.decorator_list:
        name = dec.attr if isinstance(dec, ast.Attribute) else getattr(dec, "id", "")
        if name == "abstractmethod":
            return "abstract"
    if tokens and sum(t.startswith("C:") for t in tokens) / len(tokens) >= LITERAL_DENSE_RATIO:
        return "literal-dense"
    return None


def extract_units(
    file_path: str,
    source: str,
    occurrences: Optional[OccurrenceMap],
    min_nodes: int = DEFAULT_MIN_NODES,
) -> List[FunctionUnit]:
    """Normalised units for every function in ``source`` with at least ``min_nodes`` tokens."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    units: List[FunctionUnit] = []

    def visit(node: ast.AST, enclosing: Optional[ast.ClassDef]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tokens = normalize_function(child, occurrences)
                if len(tokens) >= min_nodes:
                    units.append(
                        FunctionUnit(
                            file_path=file_path,
                            name=child.name,
                            line=child.lineno,
                            size=len(tokens),
                            tokens=tuple(tokens),
                            exact_hash=_digest(tokens),
                            shape_hash=_digest(_shape(tokens)),
                            sanctioned=_sanction(child, enclosing, tokens),
                        )
                    )
                visit(child, None)
            else:
                visit(child, enclosing)

    visit(tree, None)
    return units


# -- grouping ----------------------------------------------------------------


def _cross_file(members: Sequence[FunctionUnit]) -> bool:
    return len({m.file_path for m in members}) > 1


def hash_families(units: Sequence[FunctionUnit]) -> List[StructuralFamily]:
    """``exact`` families, then ``template`` families of the remaining shapes."""
    by_exact: Dict[str, List[FunctionUnit]] = defaultdict(list)
    by_shape: Dict[str, List[FunctionUnit]] = defaultdict(list)
    for unit in units:
        by_exact[unit.exact_hash].append(unit)
        by_shape[unit.shape_hash].append(unit)
    families = [
        StructuralFamily("exact", 1.0, sorted(group, key=lambda u: (u.file_path, u.line)))
        for group in by_exact.values()
        if _cross_file(group)
    ]
    for group in by_shape.values():
        if _cross_file(group) and len({u.exact_hash for u in group}) > 1:
            families.append(
                StructuralFamily(
                    "template", 1.0, sorted(group, key=lambda u: (u.file_path, u.line))
                )
            )
    return families


_PLACEHOLDER = re.compile(r"v\d+")


def _shingle_hashes(tokens: Sequence[str]) -> List[int]:
    """32-bit hashes of the ``SHINGLE``-token windows, placeholders unnumbered.

    Placeholder numbers follow first use, so one inserted statement would
    renumber every later local and change every downstream shingle; the
    exact hash keeps the numbers, the near tier must not.
    """
    tokens = ["v" if _PLACEHOLDER.fullmatch(t) else t for t in tokens]
    if len(tokens) < SHINGLE:
        return []
    return sorted(
        {
            int.from_bytes(
                hashlib.blake2b(
                    "\x1f".join(tokens[i : i + SHINGLE]).encode("utf-8"), digest_size=4
                ).digest(),
                "little",
            )
            for i in range(len(tokens) - SHINGLE + 1)
        }
    )


def near_pairs(
    units: Sequence[FunctionUnit],
    threshold: float = DEFAULT_NEAR_THRESHOLD,
    max_pairs: int = MAX_PAIRS,
    seed: int = 7,
) -> Tuple[List[Tuple[int, int, float]], int]:
    """Cross-file pairs with shingle Jaccard >= ``threshold`` (identical bodies excluded).

    MinHash (``NUM_PERM`` permutations, banded ``LSH_ROWS`` rows per band)
    proposes candidates; each is confirmed with the exact Jaccard. Returns
    ``(pairs, skipped_buckets)``: buckets larger than 500 members (a shape
    shared by hundreds of tiny functions) are counted, not expanded.

    Raises:
        TooManyPairsError: more than ``max_pairs`` candidate pairs.
    """
    import numpy as np

    shingles = [_shingle_hashes(u.tokens) for u in units]
    rng = np.random.default_rng(seed)
    # Universal hashing (a*x + b) mod p with x, a, b < 2**32 and p just above
    # 2**32: a*x + b < 2**64, so uint64 arithmetic is exact.
    a = rng.integers(1, 1 << 32, NUM_PERM, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, NUM_PERM, dtype=np.uint64)
    signatures = np.full((len(units), NUM_PERM), _PRIME, dtype=np.uint64)
    for i, values in enumerate(shingles):
        if values:
            x = np.asarray(values, dtype=np.uint64)[:, None]
            signatures[i] = ((a * x + b) % np.uint64(_PRIME)).min(axis=0)

    buckets: Dict[Tuple[int, bytes], List[int]] = defaultdict(list)
    for i in range(len(units)):
        if not shingles[i]:
            continue
        for band in range(NUM_PERM // LSH_ROWS):
            key = signatures[i, band * LSH_ROWS : (band + 1) * LSH_ROWS].tobytes()
            buckets[(band, key)].append(i)

    candidates: set[Tuple[int, int]] = set()
    skipped = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > 500:
            skipped += 1
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                candidates.add((members[x], members[y]))
        if len(candidates) > max_pairs:
            raise TooManyPairsError(f"over {max_pairs:,} candidate pairs; raise --min-nodes")

    sets = [set(s) for s in shingles]
    pairs = []
    for i, j in sorted(candidates):
        ui, uj = units[i], units[j]
        if ui.file_path == uj.file_path or ui.exact_hash == uj.exact_hash:
            continue
        jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
        if jac >= threshold:
            pairs.append((i, j, jac))
    return pairs, skipped


def near_families(
    units: Sequence[FunctionUnit], pairs: Sequence[Tuple[int, int, float]]
) -> List[StructuralFamily]:
    find, union = make_union_find(len(units))
    for i, j, _ in pairs:
        union(i, j)
    groups: Dict[int, List[int]] = defaultdict(list)
    best: Dict[int, float] = {}
    for i, j, jac in pairs:
        root = find(i)
        best[root] = max(best.get(root, 0.0), jac)
    for idx in sorted({i for p in pairs for i in p[:2]}):
        groups[find(idx)].append(idx)
    return [
        StructuralFamily(
            "near",
            best[root],
            sorted((units[m] for m in members), key=lambda u: (u.file_path, u.line)),
        )
        for root, members in groups.items()
    ]


def _suppression(family: StructuralFamily) -> Optional[str]:
    reasons = [m.sanctioned for m in family.members]
    if all(reasons):
        return Counter(reasons).most_common(1)[0][0]
    return None


_TIER_ORDER = {"exact": 0, "template": 1, "near": 2}


def rank(families: Sequence[StructuralFamily]) -> List[StructuralFamily]:
    """Tier first, then the most duplicated tokens."""
    return sorted(
        families,
        key=lambda f: (
            _TIER_ORDER[f.tier],
            -f.score,
            -(len(f.members) - 1) * max(m.size for m in f.members),
        ),
    )


# -- whole-repo scan ---------------------------------------------------------


def _python_files(root: Path) -> List[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        return [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", "venv", "node_modules", "build", "dist", ".victor"}
        return [
            str(p.relative_to(root))
            for p in root.rglob("*.py")
            if not skip.intersection(p.relative_to(root).parts)
        ]


def _in_scope(path: str, scope: Optional[str], include_tests: bool) -> bool:
    if scope and not path.startswith(scope):
        return False
    if not include_tests and (
        path.startswith("tests/") or any(m in "/" + path for m in TEST_PATH_MARKERS)
    ):
        return False
    return True


def scan(
    root: Path,
    scope: Optional[str] = None,
    include_tests: bool = False,
    min_nodes: int = DEFAULT_MIN_NODES,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    include_same_file: bool = False,
    scip_path: Optional[Path] = None,
) -> StructuralReport:
    """Exact, template and near families across the Python files under ``root``."""
    started = time.monotonic()
    occurrences = load_scip_occurrences(scip_path or root / SCIP_PYTHON_DUMP)
    stats: Counter[str] = Counter()
    units: List[FunctionUnit] = []
    for rel in _python_files(root):
        if not _in_scope(rel, scope, include_tests):
            continue
        try:
            source = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        occmap: Optional[OccurrenceMap] = None
        if occurrences is not None and rel in occurrences:
            occmap = fresh_occurrence_map(source.splitlines(), occurrences[rel])
            stats["scip_files" if occmap is not None else "stale_files"] += 1
        else:
            stats["ast_only_files"] += 1
        units.extend(extract_units(rel, source, occmap, min_nodes))

    families = hash_families(units)
    pairs, skipped = near_pairs(units, near_threshold)
    families += near_families(units, pairs)
    if include_same_file:
        families += _same_file_exact(units)

    kept, suppressed = [], []
    for family in families:
        family.suppressed = _suppression(family)
        (suppressed if family.suppressed else kept).append(family)

    return StructuralReport(
        families=rank(kept),
        suppressed=rank(suppressed),
        stats={
            "functions": len(units),
            "scip_files": stats["scip_files"],
            "stale_files": stats["stale_files"],
            "ast_only_files": stats["ast_only_files"],
            "scip_dump": occurrences is not None,
            "skipped_buckets": skipped,
            "seconds": round(time.monotonic() - started, 1),
            **{
                f"{tier}_families": sum(f.tier == tier for f in kept)
                for tier in ("exact", "template", "near")
            },
            "suppressed_families": len(suppressed),
        },
    )


def _same_file_exact(units: Sequence[FunctionUnit]) -> List[StructuralFamily]:
    by_key: Dict[Tuple[str, str], List[FunctionUnit]] = defaultdict(list)
    for unit in units:
        by_key[(unit.file_path, unit.exact_hash)].append(unit)
    return [
        StructuralFamily("exact", 1.0, sorted(group, key=lambda u: u.line))
        for group in by_key.values()
        if len(group) > 1
    ]


def report_to_dict(report: StructuralReport, limit: int) -> Dict[str, Any]:
    """JSON shape for ``victor graph similar --mode structural --json``."""

    def family(f: StructuralFamily) -> Dict[str, Any]:
        return {
            "tier": f.tier,
            "score": round(f.score, 4),
            "suppressed": f.suppressed,
            "files": f.files,
            "members": [
                {"file": m.file_path, "symbol": m.name, "line": m.line, "size": m.size}
                for m in f.members
            ],
        }

    return {
        "stats": report.stats,
        "families": [family(f) for f in report.families[:limit]],
        "suppressed": [family(f) for f in report.suppressed[:limit]],
    }
