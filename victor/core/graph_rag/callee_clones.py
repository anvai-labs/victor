"""Callee-set clone detection over SCIP code-intel facts.

Type-4 clone surface: two functions that call the same resolved callees are
behavioral cousins no token/AST tool can see — the callee multiset is a
compiler-resolved fingerprint (clang's CloneDetector lineage applied over
SCIP relations instead of AST hashes).

Two channels, both read-only over a SCIP dump:

- ``callee_families``: functions whose sorted resolved-reference sets are
  identical (or Jaccard >= a near threshold) form families. Exact families
  are the high-precision signal; near families need review.
- ``dead_twin_definitions``: symbols DEFINED in more than one place (or
  defined and never read anywhere) — the "two live definitions, zero
  references" dead-twin class. Reported, never auto-removed.

No score proves behavior; every family is a review candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

DEFINITION_ROLE = 0x1
READ_ACCESS_ROLE = 0x8
WRITE_ACCESS_ROLE = 0x4
TEST_CODE_ROLE = 0x20  # IsInTestCode — excluded from dead-twin reporting


@dataclass(frozen=True)
class CalleeSite:
    """One function definition: where it is and what it references."""

    symbol: str
    document: str
    start_line: int
    end_line: int
    references: tuple[str, ...]  # sorted, de-duplicated, self excluded


@dataclass
class CalleeFamily:
    """A group of functions sharing a callee fingerprint."""

    fingerprint_size: int
    jaccard: float
    members: list[CalleeSite] = field(default_factory=list)


@dataclass
class DeadTwin:
    """A definition with no read references corpus-wide (test code excluded)."""

    symbol: str
    definitions: list[tuple[str, int]]  # (document, line)
    reads: int


@dataclass
class CalleeReport:
    families: list[CalleeFamily]
    dead_twins: list[DeadTwin]
    documents_scanned: int
    definitions_scanned: int
    coverage_complete: bool


def _is_definition_obj(roles: int) -> bool:
    return bool(roles & DEFINITION_ROLE)


def _is_test_obj(roles: int) -> bool:
    return bool(roles & TEST_CODE_ROLE)


def _occurrences(doc: Any) -> Iterator[tuple[int, int, int, int, str]]:
    """Yield (start_line, start_col, end_line, roles, symbol) rows.

    Both position encodings in the wild carry line-based ranges
    (scip-python writes ``position_encoding: 0``); only malformed rows are
    skipped (mirrors structural_clones' tolerance).
    """
    for occ in doc.occurrences:
        values = occ.range
        if len(values) == 4:
            # (start_line, start_col, end_line, end_col) — multi-line
            # definition spans land here (scip-typescript and friends).
            start_line, start_col, end_line = values[0], values[1], values[2]
        elif len(values) == 3:
            # Either (line, sc, end_col) single-line (scip-python) or
            # (sl, sc, el) multi-line; both are handled — containment uses
            # line numbers only, and single-line rows clamp el to sl.
            start_line, start_col = values[0], values[1]
            end_line = values[0] if values[2] <= values[0] else values[2]
        else:
            continue
        if end_line < start_line:
            continue
        yield start_line, start_col, end_line, occ.symbol_roles, occ.symbol


def _looks_like_function(symbol: str) -> bool:
    """SCIP function/method definitions carry ``()`` in the symbol tail.

    SCIP symbols end with a trailing ``.`` (or ``#``/``:`` for types) —
    ``example .foo().`` splits with the ``()`` before the final dot, so
    strip the punctuation tail before checking.
    """
    tail = symbol.rstrip(".#:/").rsplit(".", 1)[-1]
    return tail.endswith("()")


def _definition_rows(
    rows: list[tuple[int, int, int, int, str]],
    *,
    include_tests: bool,
) -> tuple[list[tuple[int, int, str]], bool]:
    """Definition rows for a document, and whether roles were trustworthy.

    Compliant emitters set the ``Definition`` symbol role. scip-python (the
    commonest dump source) writes NO roles — there, definition is inferred
    by span: for each symbol, the occurrence with the largest span is the
    definition (a def spans the whole body; reference spans are call-site
    sized), and every other occurrence of that symbol is a reference.
    """
    if any(_is_definition_obj(roles) for _sl, _sc, _el, roles, _sym in rows):
        defs = [
            (sl, el, sym)
            for sl, _sc, el, roles, sym in rows
            if _is_definition_obj(roles) and (include_tests or not _is_test_obj(roles))
        ]
        return defs, True

    by_symbol: dict[str, list[tuple[int, int, int, int]]] = {}
    for sl, sc, el, roles, sym in rows:
        by_symbol.setdefault(sym, []).append((el - sl, sl, el, sc))
    defs: list[tuple[int, int, str]] = []
    for sym, spans in by_symbol.items():
        widest = max(spans)
        # A definition spans the whole body; an import site is one line.
        # Single-occurrence symbols keep their (possibly short) def row —
        # a def with no in-file references is still a definition.
        if len(spans) > 1 and widest[0] == 0:
            continue
        defs.append((widest[1], widest[2], sym))
    return defs, False


def callee_fingerprint_index(
    reader: Mapping[str, Any],
    *,
    include_tests: bool = False,
) -> tuple[dict[str, CalleeSite], dict[str, int]]:
    """Build the per-function reference index over a SCIP document map.

    Returns ``(sites_by_id, read_counts)`` where ``sites_by_id`` keys are
    ``symbol@document:start_line`` and ``read_counts`` counts every
    non-definition reference (counted exactly once per occurrence site,
    corpus-wide).
    """
    sites: dict[str, CalleeSite] = {}
    read_counts: dict[str, int] = {}

    for path in sorted(reader):
        rows = list(_occurrences(reader[path]))
        all_defs, roles_trusted = _definition_rows(rows, include_tests=include_tests)

        # Reference counting: every occurrence that is not a definition
        # row (matched by identity for role mode, by span for fallback).
        def_spans = {(sl, el, sym) for sl, el, sym in all_defs}
        for sl, sc, el, roles, sym in rows:
            if roles_trusted:
                if not _is_definition_obj(roles):
                    read_counts[sym] = read_counts.get(sym, 0) + 1
            else:
                if (sl, el, sym) not in def_spans:
                    read_counts[sym] = read_counts.get(sym, 0) + 1

        definitions = [(sl, el, sym) for sl, el, sym in all_defs if _looks_like_function(sym)]
        for start_line, end_line, symbol in definitions:
            refs = {
                sym
                for sl, _sc, el, _roles, sym in rows
                if sym != symbol and sl >= start_line and el <= end_line
            }
            site_id = f"{symbol}@{path}:{start_line}"
            sites[site_id] = CalleeSite(
                symbol=symbol,
                document=path,
                start_line=start_line,
                end_line=end_line,
                references=tuple(sorted(refs)),
            )

    return sites, read_counts


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _dump_is_degenerate(reader: Mapping[str, Any], sample: int = 8) -> bool:
    """True when the dump carries neither roles nor multi-line ranges.

    scip-python emits single-line ``(line, start_col, end_col)`` ranges
    with all symbol_roles zero: definition sites are indistinguishable
    from references, so callee fingerprints would report import-block
    noise. Degrade loudly instead (the CLI explains). Multi-line def
    spans (scip-typescript) or Definition roles each suffice — widest
    span / role site becomes the definition.
    """
    docs = sorted(reader)[:sample]
    if not docs:
        return True
    any_role = False
    any_multiline = False
    for path in docs:
        for sl, _sc, el, roles, _sym in _occurrences(reader[path]):
            if roles:
                any_role = True
            if el > sl:
                any_multiline = True
            if any_role or any_multiline:
                return False
    return True


def callee_families(
    reader: Mapping[str, Any],
    *,
    min_references: int = 3,
    near_threshold: float = 0.9,
    max_families: int = 50,
    include_tests: bool = False,
) -> tuple[list[CalleeFamily], dict[str, CalleeSite]]:
    """Exact-match families first, then near (Jaccard) families."""
    if _dump_is_degenerate(reader):
        return [], {}
    sites, _reads = callee_fingerprint_index(reader, include_tests=include_tests)
    eligible = {k: v for k, v in sites.items() if len(v.references) >= min_references}

    buckets: dict[tuple[str, ...], list[str]] = {}
    for site_id, site in eligible.items():
        buckets.setdefault(site.references, []).append(site_id)

    families: list[CalleeFamily] = []
    placed: set[str] = set()
    for refs, members in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        families.append(
            CalleeFamily(
                fingerprint_size=len(refs),
                jaccard=1.0,
                members=[eligible[m] for m in sorted(members)],
            )
        )
        placed.update(members)
        if len(families) >= max_families:
            return families, sites

    # Near families: greedy first-fit over remaining singletons.
    for site_id, site in sorted(eligible.items(), key=lambda kv: -len(kv[1].references)):
        if site_id in placed:
            continue
        anchor = frozenset(site.references)
        members = [site]
        for other_id, other in eligible.items():
            if other_id == site_id or other_id in placed:
                continue
            if _jaccard(anchor, frozenset(other.references)) >= near_threshold:
                members.append(other)
                placed.add(other_id)
        if len(members) >= 2:
            placed.add(site_id)
            families.append(
                CalleeFamily(
                    fingerprint_size=len(anchor),
                    jaccard=near_threshold,
                    members=sorted(members, key=lambda s: (s.document, s.start_line)),
                )
            )
            if len(families) >= max_families:
                break
    families.sort(key=lambda f: (-f.jaccard, -f.fingerprint_size, len(f.members)))
    return families, sites


def dead_twin_definitions(
    reader: Mapping[str, Any],
    *,
    include_tests: bool = False,
    max_twins: int = 100,
) -> list[DeadTwin]:
    """Definition sites with zero non-definition references corpus-wide.

    Generated code is skipped (the ``Generated`` role); test code is skipped
    unless ``include_tests``. A symbol reported here is a *candidate* dead
    twin — external consumers (other repos, macros, reflection) are outside
    the dump.
    """
    if _dump_is_degenerate(reader):
        return []
    GENERATED_ROLE = 0x10
    definitions: dict[str, list[tuple[str, int]]] = {}
    read_counts: dict[str, int] = {}
    for path in sorted(reader):
        rows = list(_occurrences(reader[path]))
        all_defs, roles_trusted = _definition_rows(rows, include_tests=include_tests)
        def_spans = {(sl, el, sym) for sl, el, sym in all_defs}
        for sl, _sc, el, roles, sym in rows:
            generated = bool(roles & GENERATED_ROLE)
            test = _is_test_obj(roles)
            if roles_trusted:
                is_def = _is_definition_obj(roles)
            else:
                is_def = (sl, el, sym) in def_spans
            if is_def:
                if generated or (test and not include_tests):
                    continue
                definitions.setdefault(sym, []).append((path, sl))
            else:
                read_counts[sym] = read_counts.get(sym, 0) + 1
    twins = [
        DeadTwin(symbol=sym, definitions=sites, reads=read_counts.get(sym, 0))
        for sym, sites in definitions.items()
        if read_counts.get(sym, 0) == 0
    ]
    twins.sort(key=lambda t: (-len(t.definitions), t.symbol))
    return twins[:max_twins]


def load_scip_documents(index_path: Path) -> Optional[Any]:
    """Lazy document reader for a SCIP dump, or None when unavailable."""
    if not index_path.is_file():
        return None
    try:
        from victor.core.graph_rag._scip import scip_pb2  # noqa: F401
        from victor.core.graph_rag.scip_documents import ScipDocumentReader

        return ScipDocumentReader(index_path)
    except (ImportError, OSError, ValueError):
        return None
