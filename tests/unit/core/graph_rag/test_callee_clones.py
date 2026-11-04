"""Tests for victor.core.graph_rag.callee_clones (SCIP callee fingerprints)."""

from __future__ import annotations

from dataclasses import dataclass, field

from victor.core.graph_rag.callee_clones import (
    CalleeSite,
    callee_families,
    callee_fingerprint_index,
    dead_twin_definitions,
)

DEFINITION = 0x1
READ = 0x8
TEST_CODE = 0x20


@dataclass
class Occ:
    range: tuple[int, ...]
    symbol: str
    symbol_roles: int = 0


@dataclass
class Doc:
    occurrences: list[Occ] = field(default_factory=list)
    position_encoding: int = 1
    text: str = ""


def _fn_def(doc: Doc, line: int, symbol: str, *, roles: int = DEFINITION) -> None:
    """A function definition spanning [line, line+2] (1-based lines)."""
    doc.occurrences.append(Occ((line, 1, line + 2), symbol, roles))


def _ref(doc: Doc, line: int, symbol: str) -> None:
    doc.occurrences.append(Occ((line, 1, line), symbol, 0))


def test_exact_family_over_identical_callee_sets() -> None:
    docs = {}
    for path in ("a.rs", "b.rs"):
        doc = Doc()
        _fn_def(doc, 10, f"example .{path}().")
        _ref(doc, 11, "std .print().")
        _ref(doc, 12, "std .format().")
        _ref(doc, 12, "std .print().")  # duplicate ref de-duplicates
        docs[path] = doc

    families, sites = callee_families(docs, min_references=2)
    exact = [f for f in families if f.jaccard == 1.0]
    assert len(exact) == 1, [f.jaccard for f in families]
    assert {m.document for m in exact[0].members} == {"a.rs", "b.rs"}
    assert exact[0].fingerprint_size == 2
    assert len(sites) == 2


def test_different_callees_do_not_family() -> None:
    docs = {}
    doc_a, doc_b = Doc(), Doc()
    _fn_def(doc_a, 5, "pkg .a().")
    _ref(doc_a, 6, "x .one().")
    _ref(doc_a, 7, "x .two().")
    _fn_def(doc_b, 9, "pkg .b().")
    _ref(doc_b, 10, "x .three().")
    _ref(doc_b, 11, "x .four().")
    docs["a.rs"] = doc_a
    docs["b.rs"] = doc_b

    families, _ = callee_families(docs, min_references=2, near_threshold=0.9)
    assert families == [] or all(f.jaccard < 1.0 for f in families)


def test_min_references_filters_small_bodies() -> None:
    doc = Doc()
    _fn_def(doc, 1, "pkg .tiny().")
    _ref(doc, 2, "x .only().")
    families, sites = callee_families({"a.rs": doc}, min_references=3)
    assert families == []
    assert len(sites) == 1  # site still indexed, just not eligible


def test_nested_definition_folds_into_enclosing() -> None:
    doc = Doc()
    doc.occurrences.append(Occ((10, 1, 16), "pkg .outer().", DEFINITION))  # spans 10-16
    _ref(doc, 11, "x .shared().")
    _fn_def(doc, 12, "pkg .inner().")  # nested, lines 12-14 fully inside
    _ref(doc, 13, "x .nested_only().")
    sites, reads = callee_fingerprint_index({"a.rs": doc})
    outer = next(s for s in sites.values() if s.symbol.endswith("outer()."))
    # inner definition's symbol is a reference site within outer's span
    assert any("inner()." in r for r in outer.references)
    assert any("nested_only()." in r for r in outer.references)
    # read counts count every non-definition occurrence exactly once
    assert reads.get("x .shared().") == 1


def test_read_counts_count_each_occurrence_once() -> None:
    doc = Doc()
    _fn_def(doc, 1, "pkg .f().")
    _ref(doc, 2, "x .shared().")
    _fn_def(doc, 10, "pkg .g().")  # second def — the ref above is outside it
    _ref(doc, 11, "x .shared().")
    _sites, reads = callee_fingerprint_index({"a.rs": doc})
    assert reads["x .shared()."] == 2


def test_dead_twin_definitions_zero_reads() -> None:
    docs = {}
    live, dead = Doc(), Doc()
    _fn_def(live, 1, "pkg .live().")
    _ref(live, 5, "pkg .live().")  # referenced elsewhere in same doc
    _fn_def(dead, 1, "pkg .dead().")  # defined, never referenced
    docs["live.rs"] = live
    docs["dead.rs"] = dead

    twins = dead_twin_definitions(docs)
    symbols = {t.symbol for t in twins}
    assert "pkg .dead()." in symbols
    assert "pkg .live()." not in symbols
    dead_twin = next(t for t in twins if t.symbol == "pkg .dead().")
    assert dead_twin.reads == 0
    assert dead_twin.definitions == [("dead.rs", 1)]


def test_test_code_and_generated_excluded_from_twins() -> None:
    doc = Doc()
    _fn_def(doc, 1, "pkg .test_helper().", roles=DEFINITION | TEST_CODE)
    _fn_def(doc, 9, "pkg .generated_fn().", roles=DEFINITION | 0x10)  # Generated
    doc.occurrences.append(Occ((5, 1, 5), "x .main()."))  # unrelated ref
    twins = dead_twin_definitions({"a.rs": doc})
    # both definitions excluded (test/generated roles); x .main(). is a bare
    # reference, not a definition, so nothing reports.
    assert {t.symbol for t in twins} == set()


def test_site_id_shape() -> None:
    doc = Doc()
    _fn_def(doc, 42, "pkg .at().")
    sites, _ = callee_fingerprint_index({"src/lib.rs": doc})
    assert "pkg .at().@src/lib.rs:42" in sites
    site = sites["pkg .at().@src/lib.rs:42"]
    assert isinstance(site, CalleeSite)
    assert site.end_line == 44
