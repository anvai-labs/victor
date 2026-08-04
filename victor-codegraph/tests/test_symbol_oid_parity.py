"""ADR-044 P2 cutover **parity ratchet**.

The canonical oid becomes the default only if it is a SAFE, non-lossy replacement for the
legacy one. This ratchet pins the invariants that gate the cutover, over a representative
multi-symbol corpus:

  1. **No collisions** — distinct symbols get distinct canonical oids (else the cutover
     would merge two symbols into one record and lose data).
  2. **Completeness** — every symbol gets a non-empty canonical oid + a legacy alias +
     a content_version.
  3. **Line-shift stability** — no symbol's canonical oid changes when lines move
     (the incremental-re-index property the whole ADR exists for).
  4. **Determinism** — re-parsing yields identical oids.

If any of these regresses, the cutover is unsafe and CI fails.
"""

from __future__ import annotations

from victor_codegraph import parse, to_proxima_records

_CORPUS = """\
import os


class Auth:
    def login(self, user):
        return self._check(user)

    def _check(self, user):
        return bool(user)


def helper(x):
    return x + 1


def helper2(x, y):
    return x + y
"""


def _nodes(records):
    return [r for r in records if "code_symbol" in r["labels"]]


def _canonical_nodes(src):
    return _nodes(
        to_proxima_records(parse(src, file_path="auth.py"), repo_graph_id="r", stable_oid=True)
    )


def test_no_collisions_distinct_symbols_distinct_oids():
    recs = _canonical_nodes(_CORPUS)
    oids = [r["oid"] for r in recs]
    assert len(recs) >= 5, "Auth, login, _check, helper, helper2 expected"
    assert len(oids) == len(set(oids)), "canonical oids must be unique per symbol (no merge)"


def test_completeness_every_symbol_has_canonical_legacy_and_version():
    for r in _canonical_nodes(_CORPUS):
        assert r["props"]["stable_oid"], "missing canonical oid"
        assert r["props"]["legacy_oid"], "missing legacy alias"
        assert r["props"]["content_version"], "missing content_version"
        assert r["oid"] == r["props"]["stable_oid"], "gated record oid must be the canonical one"


def test_line_shift_stable_for_all_symbols():
    base = {r["props"]["name"]: r["oid"] for r in _canonical_nodes(_CORPUS)}
    moved = {r["props"]["name"]: r["oid"] for r in _canonical_nodes("\n\n\n" + _CORPUS)}
    assert base == moved, "no symbol's canonical oid may change on a line shift"


def test_deterministic_reparse():
    a = {r["props"]["name"]: r["oid"] for r in _canonical_nodes(_CORPUS)}
    b = {r["props"]["name"]: r["oid"] for r in _canonical_nodes(_CORPUS)}
    assert a == b
