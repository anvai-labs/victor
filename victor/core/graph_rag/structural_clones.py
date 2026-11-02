# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Structural candidates from normalized syntax, not proven equivalent logic.

``semantic_clones`` embeds a symbol's name, signature and docstring, so it
pairs a Protocol with its implementation and misses copies whose names
drifted. This module compares function *bodies* instead:

- Each body becomes a token stream (pre-order AST walk, docstring dropped).
  Parameters keep declaration-order placeholders; lexical bindings distinguish captures,
  constants collapse to their type, and available SCIP identities refine
  global names and attributes. Without compiler evidence only their
  lexical spelling is known.
- ``exact``: identical streams in different files (copies with renamed
  variables or changed constants).
- ``template``: identical once resolved callees are erased too, while the
  callees differ (a possible abstraction for review).
- ``near``: MinHash/LSH Jaccard over 5-token shingles at or above a
  threshold (copies with a few statements added, removed or edited).

The optional subtree metric performs an exact prefix-filtered Jaccard join
over parameterized AST fingerprints. Its scores have separate calibration
from shingle Jaccard; neither metric certifies behavior or a safe refactor.

Families whose members are all sanctioned (Protocol or abstract stubs,
literal-dense data tables) are reported as suppressed with the reason,
never dropped silently. Missing or implausibly stale SCIP data falls back
to syntax and is counted; range checks alone are not a source-hash proof.
Read-only analysis; surfaced as ``victor graph similar --mode structural``.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

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
_IDENT = re.compile(r"(?:r#)?(?:[^\W\d]|_)\w*(?:\.(?:[^\W\d]|_)\w*)*")
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
    subtree_hashes: Tuple[str, ...] = ()
    language: str = "python"
    resolved_symbols: int = 0
    end_line: int = 0
    portable_hashes: Tuple[str, ...] = ()
    portable_size: int = 0
    portable_error: Optional[str] = None
    recovered_from_partial_file: bool = False
    source_kind: str = "source"
    region_index: int = 0
    source_hash: str = ""
    source_code: str = field(default="", repr=False)
    source_start: int = 0


@dataclass
class StructuralFamily:
    tier: str  # exact | template | near
    score: float
    members: List[FunctionUnit] = field(default_factory=list)
    suppressed: Optional[str] = None
    min_edge_similarity: Optional[float] = None
    matching_pairs: int = 0

    @property
    def files(self) -> List[str]:
        return sorted({m.file_path for m in self.members})


@dataclass
class StructuralReport:
    families: List[StructuralFamily]
    suppressed: List[StructuralFamily]
    stats: Dict[str, Any]
    ffi_links: List[Dict[str, Any]] = field(default_factory=list)
    units: List[FunctionUnit] = field(default_factory=list, repr=False)
    source_manifest: Dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def coverage_complete(self) -> bool:
        """Whether selected source and the requested body representation were covered."""
        return (
            self.stats.get("source_files") != 0
            and self.stats.get("selected_regions", 1) != 0
            and not self.stats.get("portable_unsupported_units", 0)
            and not any(
                self.stats.get(key)
                for key in (
                    "unavailable_languages",
                    "parse_error_files",
                    "binding_error_files",
                    "unsupported_regions",
                    "unresolved_external_regions",
                    "malformed_regions",
                    "region_limit_files",
                    "unavailable_region_extractors",
                    "unreadable_files",
                    "oversized_files",
                    "matching_budget_exceeded",
                )
            )
        )


# -- SCIP occurrence join ----------------------------------------------------


OccurrenceMap = Dict[Tuple[int, int], Tuple[int, str]]  # (line1, col) -> (length, symbol)


class _DocumentOccurrences(list[Tuple[int, int, int, str]]):
    """Keep SCIP's coordinate encoding and optional source text with its ranges."""

    def __init__(self, position_encoding: int, source_text: str) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.source_text = source_text


class _ScipDocuments(Mapping[str, List[Tuple[int, int, int, str]]]):
    """Materialize just the requested document, keeping the protobuf arena alive."""

    def __init__(self, index: Any) -> None:
        self.index = index
        self.documents = {doc.relative_path: i for i, doc in enumerate(index.documents)}

    def __iter__(self) -> Iterator[str]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, path: str) -> List[Tuple[int, int, int, str]]:
        doc = self.index.documents[self.documents[path]]
        rows = _DocumentOccurrences(doc.position_encoding, doc.text)
        symbols: dict[str, str] = {}
        for occurrence in doc.occurrences:
            values = occurrence.range
            symbol = occurrence.symbol
            symbol = symbols.setdefault(symbol, symbol)
            if len(values) == 3:
                rows.append((values[0], values[1], values[2], symbol))
            elif len(values) == 4 and values[0] == values[2]:
                rows.append((values[0], values[1], values[3], symbol))
        return rows


def load_scip_occurrences(
    index_path: Path,
) -> Optional[Mapping[str, List[Tuple[int, int, int, str]]]]:
    """Read a lazy document/range view, or None for an absent/unreadable dump.

    Every requested document retains its coordinate encoding and optional text.
    The corpus-wide protobuf is not duplicated into millions of Python tuples.
    """
    if not index_path.is_file():
        return None
    from victor.core.graph_rag.precise_index import _read_index

    index = _read_index(index_path)
    return _ScipDocuments(index) if index is not None else None


def fresh_occurrence_map(
    lines: Sequence[str], occurrences: Sequence[Tuple[int, int, int, str]]
) -> Optional[OccurrenceMap]:
    """Translate plausible ranges to UTF-8 columns, discarding invalid spans.

    Supplied document text must match. Without it, range plausibility does
    not establish freshness; the scan also checks source/dump mtimes.
    More than 20% (and over 5) invalid spans rejects the document entirely.
    """
    mapping: OccurrenceMap = {}
    bad = 0
    encoding = getattr(occurrences, "position_encoding", 1)
    if encoding not in {0, 1, 2, 3}:
        return None
    source_text = getattr(occurrences, "source_text", "")
    if source_text and source_text.splitlines() != list(lines):
        return None
    for line0, col, end, symbol in occurrences:
        if not 0 <= line0 < len(lines) or not 0 <= col < end:
            bad += 1
            continue
        line = lines[line0]
        if encoding == 0 and not line.isascii():
            bad += 1
            continue
        try:
            if encoding == 2:
                encoded = line.encode("utf-16-le")
                if end * 2 > len(encoded):
                    bad += 1
                    continue
                prefix, name = encoded[: col * 2].decode("utf-16-le"), encoded[
                    col * 2 : end * 2
                ].decode("utf-16-le")
            elif encoding == 3:
                if end > len(line):
                    bad += 1
                    continue
                prefix, name = line[:col], line[col:end]
            else:
                encoded = line.encode("utf-8")
                if end > len(encoded):
                    bad += 1
                    continue
                prefix, name = encoded[:col].decode("utf-8"), encoded[col:end].decode("utf-8")
        except UnicodeError:
            bad += 1
            continue
        if not name or not _IDENT.fullmatch(name):
            bad += 1
            continue
        # Python AST and the official tree-sitter binding use UTF-8 byte columns.
        mapping[(line0 + 1, len(prefix.encode("utf-8")))] = (len(name.encode("utf-8")), symbol)
    if bad > max(5, int(len(occurrences) * 0.2)):
        return None
    return mapping


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
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    occurrences: Optional[OccurrenceMap],
    subtrees: Optional[List[str]] = None,
    *,
    resolution_counts: Optional[Dict[str, int]] = None,
    bindings: Optional[Dict[ast.AST, int]] = None,
) -> List[str]:
    """Token stream for ``fn``'s body (see the module docstring)."""

    def field_layout(node: ast.AST) -> str:
        fields = []
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                fields.append(name + "=1")
            elif isinstance(value, list):
                holes = (
                    ":" + "".join("_" if item is None else "n" for item in value)
                    if any(item is None for item in value)
                    else ""
                )
                fields.append(f"{name}={len(value)}{holes}")
        return "|" + ";".join(fields) if fields else ""

    module = ast.Module(body=_body(fn), type_ignores=[])
    if bindings is None:
        from victor.core.graph_rag.clone_bindings import python_bindings

        # Public helper callers may supply an AST without its original source.
        # Unparsing retains the AST structure; pair nodes to restore identities.
        rendered = ast.unparse(fn)
        parsed = ast.parse(rendered)
        compiled = python_bindings(parsed, rendered)
        bindings = {
            old: compiled[new]
            for old, new in zip(ast.walk(fn), ast.walk(parsed.body[0]))
            if new in compiled
        }
    declared = {
        bindings[n]
        for n in ast.walk(fn)
        if n in bindings
        and (
            isinstance(
                n, (ast.arg, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)
            )
            or isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
        )
    }
    own_scope = f"{fn.name}()."
    placeholders: Dict[str, str] = {}
    args = fn.args
    parameters = [*args.posonlyargs, *args.args]
    if args.vararg is not None:
        parameters.append(args.vararg)
    parameters.extend(args.kwonlyargs)
    if args.kwarg is not None:
        parameters.append(args.kwarg)
    next_local = len(parameters)

    def placeholder(key: str) -> str:
        nonlocal next_local
        if key not in placeholders:
            placeholders[key] = f"v{next_local}"
            next_local += 1
        return placeholders[key]

    def resolve(key: Tuple[int, int], name: str) -> Optional[str]:
        # only a range spanning exactly this name: a stale or shifted dump misses
        hit = occurrences.get(key) if occurrences else None
        if hit is not None and hit[0] == len(name.encode("utf-8")):
            if resolution_counts is not None:
                resolution_counts["resolved_symbols"] = (
                    resolution_counts.get("resolved_symbols", 0) + 1
                )
            return hit[1]
        return None

    for i, parameter in enumerate(parameters):
        label = f"v{i}"
        if parameter in bindings:
            placeholders["binding:" + str(bindings[parameter])] = label
        symbol = resolve((parameter.lineno, parameter.col_offset), parameter.arg)
        if symbol is not None:
            placeholders[symbol] = label

    tokens: List[str] = [f"parameters:{len(parameters)}"]
    walk = list(_preorder(module))
    for node in walk:
        if node is module:
            continue
        if isinstance(node, ast.Name):
            symbol = resolve((node.lineno, node.col_offset), node.id)
            binding = bindings.get(node)
            if binding is not None and binding in declared:
                tokens.append(placeholder("binding:" + str(binding)))
            elif binding is not None:
                # An enclosing declaration outside this unit has no local
                # anchor. Retain its context instead of alpha-erasing it.
                tokens.append("G:free:" + (symbol or node.id))
            elif symbol is not None:
                if symbol.startswith("local ") or own_scope in symbol:
                    tokens.append(placeholder(symbol))
                else:
                    tokens.append("G:" + symbol)
            else:
                tokens.append("G:" + node.id)
        elif isinstance(node, ast.Attribute):
            symbol = None
            if node.end_lineno is not None and node.end_col_offset is not None:
                symbol = resolve(
                    (node.end_lineno, node.end_col_offset - len(node.attr.encode("utf-8"))),
                    node.attr,
                )
            target = symbol or node.attr
            tokens.append("A:" + target)
        elif isinstance(node, ast.Constant):
            tokens.append("C:" + type(node.value).__name__)
        elif isinstance(node, ast.arg):
            tokens.append(placeholder("binding:" + str(bindings[node])))
        elif isinstance(node, ast.keyword):
            tokens.append("G:keyword:" + (node.arg or "**"))
        elif (
            isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)
            )
            and node in bindings
        ):
            # Anchor named nested declarations before their call sites. A
            # first-use label alone erases which helper is called.
            tokens.append(
                type(node).__name__
                + field_layout(node)
                + ":"
                + placeholder("binding:" + str(bindings[node]))
            )
        elif isinstance(node, ast.alias):
            tokens.append("G:import:" + node.name)
        elif isinstance(node, ast.ImportFrom):
            tokens.append("G:import-from:" + "." * node.level + (node.module or ""))
        else:
            tokens.append(type(node).__name__ + field_layout(node))
    if subtrees is not None:
        from victor.core.graph_rag.ast_fingerprints import NormalizedNode, subtree_fingerprints

        normalized = {}
        for node, label in reversed(list(zip(walk[1:], tokens[1:]))):
            children = tuple(
                (
                    NormalizedNode("field:" + name, (normalized[value],))
                    if isinstance(value, ast.AST)
                    else NormalizedNode(
                        "field:" + name,
                        tuple(
                            (
                                normalized[item]
                                if isinstance(item, ast.AST)
                                else NormalizedNode("omitted")
                            )
                            for item in value
                            if isinstance(item, ast.AST) or item is None
                        ),
                    )
                )
                for name, value in ast.iter_fields(node)
                if isinstance(value, ast.AST) or isinstance(value, list)
            )
            if (
                isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)
                )
                and node in bindings
            ):
                kind, binding = label.split(":", 1)
                normalized[node] = NormalizedNode(kind, (NormalizedNode(binding), *children))
                continue
            normalized[node] = NormalizedNode(label, children)
        body = NormalizedNode("Module", tuple(normalized[c] for c in ast.iter_child_nodes(module)))
        tree = NormalizedNode(
            tokens[0],
            (
                NormalizedNode(
                    "parameters", tuple(NormalizedNode(f"v{i}") for i in range(len(parameters)))
                ),
                body,
            ),
        )
        subtrees.extend(subtree_fingerprints(tree, "python"))
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
    with_subtrees: bool = False,
    strict_parse: bool = False,
    with_portable: bool = False,
    retain_source: bool = False,
) -> List[FunctionUnit]:
    """Normalised units for every function in ``source`` with at least ``min_nodes`` tokens."""
    try:
        tree = ast.parse(source)
        from victor.core.graph_rag.clone_bindings import python_bindings

        bindings = python_bindings(tree, source)
    except SyntaxError:
        if strict_parse:
            raise
        return []
    units: List[FunctionUnit] = []
    source_bytes = source.encode("utf-8") if retain_source else b""
    line_offsets = [0, *(i + 1 for i, value in enumerate(source_bytes) if value == 10)]

    def visit(node: ast.AST, enclosing: Optional[ast.ClassDef]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                subtrees: List[str] = []
                counts = {"resolved_symbols": 0}
                tokens = normalize_function(
                    child,
                    occurrences,
                    subtrees if with_subtrees else None,
                    resolution_counts=counts,
                    bindings=bindings,
                )
                if len(tokens) >= min_nodes:
                    portable_hashes: Tuple[str, ...] = ()
                    portable_size = 0
                    portable_error: Optional[str] = None
                    if with_portable:
                        from victor.core.graph_rag.portable_clones import (
                            python_body,
                            UnsupportedOperation,
                        )

                        try:
                            portable = python_body(child, bindings=bindings)
                            portable_hashes, portable_size = portable.fingerprints, portable.size
                        except (UnsupportedOperation, RecursionError) as exc:
                            portable_error = str(exc)
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
                            subtree_hashes=tuple(subtrees),
                            resolved_symbols=counts["resolved_symbols"],
                            end_line=child.end_lineno or child.lineno,
                            portable_hashes=portable_hashes,
                            portable_size=portable_size,
                            portable_error=portable_error,
                            source_code=(
                                source_bytes[
                                    line_offsets[child.lineno - 1]
                                    + child.col_offset : line_offsets[
                                        (child.end_lineno or child.lineno) - 1
                                    ]
                                    + (child.end_col_offset or 0)
                                ].decode("utf-8")
                                if retain_source
                                else ""
                            ),
                            source_start=(
                                line_offsets[child.lineno - 1] + child.col_offset
                                if retain_source
                                else 0
                            ),
                        )
                    )
                visit(child, None)
            else:
                visit(child, enclosing)

    visit(tree, None)
    return units


def _recover_python_units(
    path: str,
    source: str,
    min_nodes: int,
    *,
    with_subtrees: bool,
    with_portable: bool,
    retain_source: bool,
) -> List[FunctionUnit]:
    """Recover clean callable fragments, without inventing enclosing bindings."""
    from bisect import bisect_right
    from victor.core.graph_rag.language_parsers import create_parser

    parser = create_parser("python")
    if parser is None:
        return []
    data = source.encode("utf-8")
    starts = [0, *(i + 1 for i, value in enumerate(data) if value == 10)]
    tree = parser.parse(data)
    result = []

    def visit(node: Any) -> None:
        if node.type == "function_definition":
            if node.has_error:
                return
            fragment = data[node.start_byte : node.end_byte].decode("utf-8")
            try:
                recovered = extract_units(
                    path,
                    fragment,
                    None,
                    min_nodes,
                    with_subtrees=with_subtrees,
                    strict_parse=True,
                    with_portable=with_portable,
                    retain_source=retain_source,
                )
            except (SyntaxError, ValueError, RecursionError):
                # Isolated nonlocal declarations need missing outer context.
                return
            row = bisect_right(starts, node.start_byte) - 1
            result.extend(
                replace(
                    unit,
                    line=unit.line + row,
                    end_line=unit.end_line + row,
                    recovered_from_partial_file=True,
                    source_start=node.start_byte + unit.source_start,
                )
                for unit in recovered
            )
            return
        for child in node.named_children:
            visit(child)

    try:
        visit(tree.root_node)
    except RecursionError:
        pass  # The owning file already carries incomplete parse coverage.
    return result


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
    include_same_file: bool = False,
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
                left, right = units[members[x]], units[members[y]]
                if left.language != right.language:
                    continue
                if (
                    left.file_path == right.file_path and not include_same_file
                ) or left.exact_hash == right.exact_hash:
                    continue
                candidates.add((members[x], members[y]))
                if len(candidates) > max_pairs:
                    raise TooManyPairsError(
                        f"over {max_pairs:,} candidate pairs; raise --min-nodes"
                    )

    sets = [set(s) for s in shingles]
    pairs = []
    for i, j in sorted(candidates):
        ui, uj = units[i], units[j]
        if ui.language != uj.language:
            continue
        if (
            ui.file_path == uj.file_path and not include_same_file
        ) or ui.exact_hash == uj.exact_hash:
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
    weakest: Dict[int, float] = {}
    counts: Counter[int] = Counter()
    for i, j, jac in pairs:
        root = find(i)
        best[root] = max(best.get(root, 0.0), jac)
        weakest[root] = min(weakest.get(root, 1.0), jac)
        counts[root] += 1
    for idx in sorted({i for p in pairs for i in p[:2]}):
        groups[find(idx)].append(idx)
    return [
        StructuralFamily(
            "near",
            best[root],
            sorted((units[m] for m in members), key=lambda u: (u.file_path, u.line)),
            min_edge_similarity=weakest[root],
            matching_pairs=counts[root],
        )
        for root, members in groups.items()
    ]


def _subtree_pairs(
    units: Sequence[FunctionUnit], threshold: float, include_same_file: bool
) -> List[Tuple[int, int, float]]:
    """Score identical fingerprint sets once; retain valid connecting edges.

    For each matching group pair, a spanning forest of the complete bipartite
    graph preserves its components without expanding every duplicate pair.
    Same-file edges remain excluded when requested, including two-file splits.
    """
    from victor.core.graph_rag.ast_fingerprints import jaccard_pairs

    grouped: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for i, unit in enumerate(units):
        grouped[(unit.language, unit.subtree_hashes)].append(i)
    groups = list(grouped.values())
    pairs = []
    for a, b, score in jaccard_pairs([units[g[0]].subtree_hashes for g in groups], threshold):
        remaining: List[Dict[str, List[int]]] = [defaultdict(list), defaultdict(list)]
        for side, group in enumerate((groups[a], groups[b])):
            for i in group:
                remaining[side][units[i].file_path].append(i)
        queue: deque[Tuple[int, int]] = deque()
        while remaining[0] or remaining[1] or queue:
            if not queue:
                side = 0 if remaining[0] else 1
                file = next(iter(remaining[side]))
                i = remaining[side][file].pop()
                if not remaining[side][file]:
                    del remaining[side][file]
                queue.append((side, i))
            side, i = queue.popleft()
            other = remaining[1 - side]
            for file in list(other):
                if not include_same_file and file == units[i].file_path:
                    continue
                for j in other.pop(file):
                    queue.append((1 - side, j))
                    pairs.append((min(i, j), max(i, j), score))
                    if len(pairs) > MAX_PAIRS:
                        raise TooManyPairsError(
                            "AST connecting edge budget exceeded; raise --min-nodes"
                        )
    return pairs


def _suppression(family: StructuralFamily) -> Optional[str]:
    reasons = [m.sanctioned for m in family.members]
    if all(reasons):
        return Counter(reasons).most_common(1)[0][0]
    return None


_TIER_ORDER = {"exact": 0, "template": 1, "near": 2, "cross_language": 3}


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


def _source_files(root: Path, include_docs: bool = False) -> List[str]:
    from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES
    from victor.core.graph_rag.source_regions import EMBEDDED_SUFFIXES, DOC_SUFFIXES

    suffixes = set(SOURCE_LANGUAGES) | EMBEDDED_SUFFIXES | (DOC_SUFFIXES if include_docs else set())

    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        return [p for p in out.split("\0") if p and Path(p).suffix.lower() in suffixes]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", "venv", "node_modules", "build", "dist", ".victor"}
        files = []
        for directory, subdirs, names in os.walk(root):
            subdirs[:] = sorted(name for name in subdirs if name not in skip)
            for name in sorted(names):
                path = Path(directory) / name
                if path.suffix.lower() in suffixes:
                    files.append(path.relative_to(root).as_posix())
        return files


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
    near_metric: str = "shingle",
    max_file_bytes: int = 1_000_000,
    progress: Optional[Callable[[str, int, int], None]] = None,
    languages: Optional[set[str]] = None,
    cross_language: bool = False,
    include_docs: bool = False,
    retain_source: bool = False,
    collect_only: bool = False,
    max_units: int | None = None,
) -> StructuralReport:
    """Normalized body candidates across supported source languages.

    Available per-language SCIP dumps refine syntax-only name resolution.
    Neither tier establishes behavioral equivalence.
    """
    if not root.is_dir():
        raise ValueError(f"Project path is not a directory: {root}")
    if near_metric not in {"shingle", "subtree"}:
        raise ValueError("Unknown AST metric; use subtree or shingle")
    if (
        min_nodes < 1
        or max_file_bytes < 1
        or not math.isfinite(near_threshold)
        or not 0 < near_threshold <= 1
    ):
        raise ValueError("Structural size budgets and threshold must be positive and finite")
    started = time.monotonic()
    from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES, create_parser
    from victor.core.graph_rag.precise_providers import providers_for_files
    from victor.core.graph_rag.exclude_patterns import UNIVERSAL_EXCLUDE_PATTERNS, is_path_excluded
    from victor.core.graph_rag.source_reader import project_source_reader

    if languages is not None and not languages <= set(SOURCE_LANGUAGES.values()):
        raise ValueError(
            f"Unsupported structural languages: {sorted(languages - set(SOURCE_LANGUAGES.values()))}"
        )
    with project_source_reader(root) as read_source:
        selected = [
            rel
            for rel in _source_files(root, include_docs=include_docs)
            if _in_scope(rel, scope, include_tests)
            and (
                languages is None
                or Path(rel).suffix.lower() not in SOURCE_LANGUAGES
                or SOURCE_LANGUAGES[Path(rel).suffix.lower()] in languages
            )
        ]
        files = [
            rel
            for rel in selected
            if not is_path_excluded(root / rel, root, UNIVERSAL_EXCLUDE_PATTERNS)
        ]
        paths = (
            []
            if collect_only
            else (
                [scip_path]
                if scip_path
                else [
                    root / ".victor" / "precise" / f"{name}.scip"
                    for name in providers_for_files(files)
                ]
            )
        )
        documents: dict[str, Mapping[str, List[Tuple[int, int, int, str]]]] = {}
        loaded_dumps = []
        dump_times: dict[str, int] = {}
        for dump in paths:
            loaded = load_scip_occurrences(dump)
            if loaded is not None:
                documents.update(dict.fromkeys(loaded, loaded))
                dump_times.update({rel: dump.stat().st_mtime_ns for rel in loaded})
                loaded_dumps.append(dump.name)
        stats: Counter[str] = Counter()
        unavailable: list[str] = []
        parsers: dict[str, Any] = {}
        units: List[FunctionUnit] = []
        source_hashes: Dict[str, str] = {}
        source_manifest: Dict[str, str] = {}
        binding_facts: List[Any] = []
        source_read_errors: List[Dict[str, str]] = []
        retained_bytes = 0
        for file_index, rel in enumerate(files):
            if progress is not None:
                progress(rel, file_index, len(files))
            try:
                data, source_stat = read_source(rel, max_file_bytes)
                if source_stat.st_size > max_file_bytes:
                    stats["oversized_files"] += 1
                    continue
                if len(data) > max_file_bytes:
                    stats["oversized_files"] += 1
                    continue
                source = data.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                stats["unreadable_files"] += 1
                source_read_errors.append({"file": rel, "reason": str(exc)})
                continue
            if cross_language:
                source_hashes[rel] = hashlib.sha256(
                    source.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
                ).hexdigest()
            occmap: Optional[OccurrenceMap] = None
            if rel in documents:
                occurrences = documents[rel][rel]
                source_text = getattr(occurrences, "source_text", "")
                if source_text or source_stat.st_mtime_ns <= dump_times[rel]:
                    occmap = fresh_occurrence_map(source.splitlines(), occurrences)
                stats["scip_files" if occmap is not None else "stale_files"] += 1
            else:
                stats["ast_only_files"] += 1
            from victor.core.graph_rag.source_regions import SourceRegion, source_regions

            suffix = Path(rel).suffix.lower()
            host_hash = hashlib.sha256(data).hexdigest()
            source_manifest[rel] = host_hash
            if suffix in SOURCE_LANGUAGES:
                regions = [SourceRegion(source, SOURCE_LANGUAGES[suffix])]
            else:
                regions, region_stats = source_regions(source, suffix)
                stats.update(region_stats)
                stats["embedded_regions"] += len(regions)
                source_hashes.pop(rel, None)
            file_parse_error = False
            file_binding_error = False
            for region in regions:
                language = region.language
                if languages is not None and language not in languages:
                    stats["filtered_regions"] += 1
                    continue
                stats[f"{language}_files"] += 1
                stats["selected_regions"] += 1
                compiler = occmap if region.kind == "source" else None
                extracted = []
                try:
                    if language == "python":
                        extracted = extract_units(
                            rel,
                            region.source,
                            compiler,
                            min_nodes,
                            with_subtrees=near_metric == "subtree" and not collect_only,
                            strict_parse=True,
                            with_portable=cross_language,
                            retain_source=retain_source,
                        )
                    else:
                        from victor.core.graph_rag.treesitter_clones import extract_tree_units

                        if language in unavailable:
                            continue
                        if language not in parsers:
                            parser = create_parser(language)
                            if parser is None:
                                unavailable.append(language)
                                continue
                            parsers[language] = parser
                        diagnostics: Dict[str, int] = {}
                        extracted = extract_tree_units(
                            rel,
                            region.source,
                            parsers[language],
                            min_nodes,
                            include_tests,
                            language,
                            compiler,
                            with_subtrees=near_metric == "subtree" and not collect_only,
                            with_portable=cross_language,
                            binding_facts=(
                                binding_facts
                                if cross_language and region.kind == "source"
                                else None
                            ),
                            diagnostics=diagnostics,
                            retain_source=retain_source,
                        )
                        file_parse_error |= bool(diagnostics.get("parse_error_files"))
                except SyntaxError:
                    file_parse_error = True
                    if language == "python":
                        extracted = _recover_python_units(
                            rel,
                            region.source,
                            min_nodes,
                            with_subtrees=near_metric == "subtree",
                            with_portable=cross_language,
                            retain_source=retain_source,
                        )
                except RecursionError:
                    file_parse_error = True
                except ValueError:
                    file_binding_error = True
                retained_bytes += sum(len(unit.source_code.encode("utf-8")) for unit in extracted)
                if (
                    max_units is not None and len(units) + len(extracted) > max_units
                ) or retained_bytes > 128 * 1024 * 1024:
                    raise ValueError(
                        "Source analysis exceeds its function/source retention budget; narrow --scope"
                    )
                units.extend(
                    replace(
                        unit,
                        line=unit.line + region.line_offset,
                        end_line=unit.end_line + region.line_offset,
                        source_kind=region.kind,
                        region_index=region.index,
                        source_hash=host_hash,
                    )
                    for unit in extracted
                )
            stats["parse_error_files"] += file_parse_error
            stats["binding_error_files"] += file_binding_error
            if file_parse_error or file_binding_error:
                source_hashes.pop(rel, None)

        families = [] if collect_only else hash_families(units)
        analysis_errors: List[str] = []
        skipped = 0
        try:
            if collect_only:
                pairs = []
            elif near_metric == "subtree":
                pairs = _subtree_pairs(units, near_threshold, include_same_file)
            else:
                pairs, skipped = near_pairs(
                    units, near_threshold, include_same_file=include_same_file
                )
            families += near_families(units, pairs)
        except TooManyPairsError as exc:
            stats["matching_budget_exceeded"] += 1
            analysis_errors.append(str(exc))
        ffi_links: List[Dict[str, Any]] = []
        if cross_language and not collect_only:
            from victor.core.graph_rag.polyglot_clones import (
                cross_language_families,
                ffi_relationships,
            )

            try:
                families += cross_language_families(units, near_threshold)
            except TooManyPairsError as exc:
                stats["matching_budget_exceeded"] += 1
                analysis_errors.append(str(exc))
            ffi_links, ffi_stats = ffi_relationships(root, units, source_hashes, binding_facts)
            stats.update(ffi_stats)
            stats["ffi_relationships"] = len(ffi_links)
        if include_same_file and not collect_only:
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
                "source_files": len(files),
                "coverage_scope": "selected_supported_frontends_and_script_regions",
                "include_docs": include_docs,
                "selected_regions": stats["selected_regions"],
                **{key: value for key, value in stats.items() if "region" in key},
                "normalization_schema": "normalized-body-v4-field-roles",
                "binding_support": {
                    lang: (
                        "compiler-symbol-tables"
                        if lang == "python"
                        else (
                            "lexical-subset"
                            if lang in {"rust", "javascript", "typescript", "tsx"}
                            else "declaration-subset"
                        )
                    )
                    for lang in sorted(set(u.language for u in units))
                },
                "excluded_artifact_files": len(selected) - len(files),
                "near_metric": near_metric,
                "cross_language": cross_language,
                **(_portable_stats(units) if cross_language else {}),
                **{key: value for key, value in stats.items() if key.startswith("ffi_")},
                "rust_files": stats["rust_files"],
                "python_files": stats["python_files"],
                "unavailable_languages": unavailable,
                "parse_error_files": stats["parse_error_files"],
                "binding_error_files": stats["binding_error_files"],
                "recovered_units": sum(u.recovered_from_partial_file for u in units),
                "unreadable_files": stats["unreadable_files"],
                "source_read_errors": source_read_errors,
                "oversized_files": stats["oversized_files"],
                "languages": {
                    lang: stats[f"{lang}_files"]
                    for lang in sorted(set(SOURCE_LANGUAGES.values()))
                    if stats[f"{lang}_files"]
                },
                "resolved_symbols": sum(u.resolved_symbols for u in units),
                "scip_dumps": loaded_dumps,
                "scip_freshness": "source-text-when-present-otherwise-mtime-and-range-heuristic",
                "scip_files": stats["scip_files"],
                "stale_files": stats["stale_files"],
                "ast_only_files": stats["ast_only_files"],
                "scip_dump": bool(loaded_dumps),
                "skipped_buckets": skipped,
                "matching_budget_exceeded": stats["matching_budget_exceeded"],
                "analysis_errors": analysis_errors,
                "seconds": round(time.monotonic() - started, 1),
                **{
                    f"{tier}_families": sum(f.tier == tier for f in kept)
                    for tier in ("exact", "template", "near", "cross_language")
                },
                "suppressed_families": len(suppressed),
            },
            ffi_links=ffi_links,
            units=units,
            source_manifest=source_manifest,
        )


def _portable_stats(units: Sequence[FunctionUnit]) -> Dict[str, Any]:
    from victor.core.graph_rag.portable_clones import MIN_PORTABLE_NODES, PORTABLE_SCHEMA

    unsupported = sum(u.portable_error is not None for u in units)
    return {
        "portable_schema": PORTABLE_SCHEMA,
        "portable_units": sum(u.portable_size >= MIN_PORTABLE_NODES for u in units),
        "portable_unsupported_units": unsupported,
        "portable_small_units": sum(
            not u.portable_error and u.portable_size < MIN_PORTABLE_NODES for u in units
        ),
        "portable_unsupported_reasons": dict(
            Counter(u.portable_error for u in units if u.portable_error)
        ),
        "portable_coverage_complete": not unsupported,
    }


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
            "score_basis": (
                "maximum_matching_edge"
                if f.tier in {"near", "cross_language"}
                else "normalized_hash_equality"
            ),
            "evidence": (
                "portable_operation_shape" if f.tier == "cross_language" else "normalized_syntax"
            ),
            "min_edge_similarity": f.min_edge_similarity,
            "matching_edges": f.matching_pairs,
            "suppressed": f.suppressed,
            "files": f.files,
            "members": [
                {
                    "file": m.file_path,
                    "symbol": m.name,
                    "line": m.line,
                    "size": m.size,
                    "language": m.language,
                    "resolved_symbols": m.resolved_symbols,
                    "end_line": m.end_line,
                    "source_kind": m.source_kind,
                    "region_index": m.region_index,
                    "source_hash": m.source_hash,
                    "recovered_from_partial_file": m.recovered_from_partial_file,
                    **(
                        {"portable_size": m.portable_size, "portable_error": m.portable_error}
                        if report.stats.get("cross_language")
                        else {}
                    ),
                }
                for m in f.members
            ],
        }

    return {
        "evidence": "normalized_syntax",
        "coverage_complete": report.coverage_complete,
        "equivalence_verified": False,
        "near_grouping": "connected_components",
        "stats": report.stats,
        "families": [family(f) for f in report.families[:limit]],
        "suppressed": [family(f) for f in report.suppressed[:limit]],
        "ffi_links": report.ffi_links[:limit],
    }
