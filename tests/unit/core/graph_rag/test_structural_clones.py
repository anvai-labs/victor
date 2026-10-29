# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Structural clone detection: normalisation, tiers, suppression, SCIP join."""

from __future__ import annotations

import ast
import subprocess
import textwrap
from pathlib import Path

import pytest

from victor.core.graph_rag import structural_clones as sc
from victor.core.graph_rag.semantic_clones import TooManyPairsError

PKG = "scip-python python victor 0.1.0 "

BODY_A = '''
def total(items, scale):
    """Sum scaled values."""
    acc = 0
    for item in items:
        if item.enabled and item.weight > 0:
            acc += item.weight * scale
        else:
            acc -= 1
    result = round(acc, 2)
    return {"total": result, "count": len(items), "scale": scale}
'''

# Renamed every local and changed the constants: same logic.
BODY_A_RENAMED = """
def summed(rows, factor):
    running = 10
    for row in rows:
        if row.enabled and row.weight > 5:
            running += row.weight * factor
        else:
            running -= 3
    out = round(running, 4)
    return {"sum": out, "n": len(rows), "f": factor}
"""

# Same shape but calls a different function (floor instead of round).
BODY_A_OTHER_CALLEE = BODY_A_RENAMED.replace("round(", "floor(")

# One extra statement and one changed condition: an edited copy.
BODY_A_EDITED = """
def summed(rows, factor):
    running = 0
    seen = set()
    for row in rows:
        if row.enabled and row.weight >= 0:
            running += row.weight * factor
        else:
            running -= 1
    out = round(running, 2)
    return {"total": out, "count": len(rows), "scale": factor}
"""

UNRELATED = """
def build(config, logger):
    parts = []
    for key in sorted(config):
        value = config[key]
        while value and value.parent is not None:
            value = value.parent
        parts.append((key, value))
    logger.info("built %d", len(parts))
    return dict(parts)
"""


def _units(source: str, path: str, occ=None, min_nodes: int = 10):
    return sc.extract_units(path, textwrap.dedent(source), occ, min_nodes)


def _occurrences(source: str, symbols: dict[str, str]) -> sc.OccurrenceMap:
    """Hand-built SCIP-style occurrence map: each Name/attribute token -> symbol."""
    tree = ast.parse(textwrap.dedent(source))
    out: sc.OccurrenceMap = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in symbols:
            out[(node.lineno, node.col_offset)] = (len(node.id), symbols[node.id])
        elif isinstance(node, ast.Attribute) and node.attr in symbols:
            key = (node.end_lineno, node.end_col_offset - len(node.attr))
            out[key] = (len(node.attr), symbols[node.attr])
    return out


class TestNormalisation:
    def test_renamed_locals_and_constants_hash_identically(self):
        (a,) = _units(BODY_A, "a.py")
        (b,) = _units(BODY_A_RENAMED, "b.py")
        assert a.exact_hash == b.exact_hash

    def test_docstring_is_ignored(self):
        (a,) = _units(BODY_A, "a.py")
        (b,) = _units(BODY_A.replace('"""Sum scaled values."""', '"""Other."""'), "b.py")
        assert a.exact_hash == b.exact_hash

    def test_different_callee_keeps_shape_but_not_exact(self):
        (a,) = _units(BODY_A, "a.py")
        (b,) = _units(BODY_A_OTHER_CALLEE, "b.py")
        assert a.exact_hash != b.exact_hash
        assert a.shape_hash == b.shape_hash

    def test_scip_symbols_separate_same_spelling_different_target(self):
        """`.get` on a registry and on a cache are different callees."""
        src = "def f(k):\n    v = store.get(k)\n    return v.value + v.value * 2 + len(k)\n"
        reg = _occurrences(src, {"store": "local 0", "get": PKG + "`m`/Registry#get()."})
        cache = _occurrences(src, {"store": "local 0", "get": PKG + "`m`/Cache#get()."})
        (a,) = _units(src, "a.py", reg, min_nodes=5)
        (b,) = _units(src, "b.py", cache, min_nodes=5)
        (plain_a,) = _units(src, "a.py", None, min_nodes=5)
        (plain_b,) = _units(src, "b.py", None, min_nodes=5)
        assert plain_a.exact_hash == plain_b.exact_hash  # AST alone cannot tell them apart
        assert a.exact_hash != b.exact_hash
        assert a.shape_hash == b.shape_hash

    def test_scip_local_symbol_becomes_placeholder(self):
        src = "def f(k):\n    v = helper(k)\n    return v + v * 2 + k\n"
        occ = _occurrences(src, {"v": "local 3", "helper": PKG + "`m`/helper()."})
        tokens = sc.normalize_function(ast.parse(src).body[0], occ)
        assert "G:`m`/helper()." in tokens
        assert not any("v" == t or t == "G:v" for t in tokens)

    def test_self_member_drops_the_class_prefix(self):
        src_a = "class A:\n    def f(self):\n        return self.load() + self.load() * 2\n"
        src_b = src_a.replace("class A", "class B")
        occ_a = _occurrences(src_a, {"load": PKG + "`m`/A#load()."})
        occ_b = _occurrences(src_b, {"load": PKG + "`m`/B#load()."})
        (a,) = _units(src_a, "a.py", occ_a, min_nodes=5)
        (b,) = _units(src_b, "b.py", occ_b, min_nodes=5)
        assert a.exact_hash == b.exact_hash

    def test_small_functions_are_skipped(self):
        assert _units("def f():\n    return 1\n", "a.py", min_nodes=10) == []


class TestFamilies:
    def test_exact_family_is_cross_file_only(self):
        units = _units(BODY_A, "a.py") + _units(BODY_A_RENAMED, "b.py")
        families = sc.hash_families(units)
        assert [f.tier for f in families] == ["exact"]
        assert families[0].files == ["a.py", "b.py"]
        same_file = _units(BODY_A + BODY_A_RENAMED, "a.py")
        assert sc.hash_families(same_file) == []

    def test_template_family_when_only_callees_differ(self):
        units = _units(BODY_A, "a.py") + _units(BODY_A_OTHER_CALLEE, "b.py")
        families = sc.hash_families(units)
        assert [f.tier for f in families] == ["template"]

    def test_edited_copy_is_near_not_exact(self):
        units = _units(BODY_A, "a.py") + _units(BODY_A_EDITED, "b.py") + _units(UNRELATED, "c.py")
        assert sc.hash_families(units) == []
        pairs, skipped = sc.near_pairs(units, threshold=0.5)
        assert skipped == 0
        assert [(units[i].file_path, units[j].file_path) for i, j, _ in pairs] == [("a.py", "b.py")]
        (fam,) = sc.near_families(units, pairs)
        assert fam.tier == "near" and 0.5 <= fam.score < 1.0

    def test_unrelated_logic_is_not_grouped(self):
        units = _units(BODY_A, "a.py") + _units(UNRELATED, "c.py")
        assert sc.hash_families(units) == []
        assert sc.near_pairs(units, threshold=0.3)[0] == []

    def test_identical_bodies_are_not_double_counted_as_near(self):
        units = _units(BODY_A, "a.py") + _units(BODY_A_RENAMED, "b.py")
        assert sc.near_pairs(units, threshold=0.5)[0] == []

    def test_candidate_budget(self):
        units = [u for i in range(6) for u in _units(BODY_A, f"f{i}.py")]
        units += [u for i in range(6) for u in _units(BODY_A_EDITED, f"g{i}.py")]
        with pytest.raises(TooManyPairsError):
            sc.near_pairs(units, threshold=0.5, max_pairs=3)


class TestSuppression:
    def test_protocol_stub_pair_is_suppressed(self):
        proto = "from typing import Protocol\n" "class P(Protocol):\n" + textwrap.indent(
            textwrap.dedent(BODY_A), "    "
        )
        units = _units(proto, "p.py") + _units(proto.replace("class P", "class Q"), "q.py")
        (fam,) = sc.hash_families(units)
        assert sc._suppression(fam) == "protocol"

    def test_mixed_family_is_not_suppressed(self):
        proto = "class P(Protocol):\n" + textwrap.indent(textwrap.dedent(BODY_A), "    ")
        units = _units(proto, "p.py") + _units(BODY_A_RENAMED, "b.py")
        (fam,) = sc.hash_families(units)
        assert sc._suppression(fam) is None

    def test_literal_dense_data_tables_are_suppressed(self):
        table = "def cfg():\n    return {" + ", ".join(f"'k{i}': {i}" for i in range(20)) + "}\n"
        units = _units(table, "a.py") + _units(table.replace("cfg", "other"), "b.py")
        (fam,) = sc.hash_families(units)
        assert sc._suppression(fam) == "literal-dense"

    def test_abstract_methods_are_sanctioned(self):
        src = "class A:\n    @abstractmethod\n" + textwrap.indent(textwrap.dedent(BODY_A), "    ")
        (unit,) = _units(src, "a.py")
        assert unit.sanctioned == "abstract"


class TestScipJoin:
    def test_fresh_occurrences_map_by_one_based_line(self):
        lines = ["def f(x):", "    return helper(x)"]
        occ = [(1, 11, 17, PKG + "`m`/helper().")]
        assert sc.fresh_occurrence_map(lines, occ) == {(2, 11): (6, PKG + "`m`/helper().")}

    def test_stale_document_is_rejected(self):
        lines = ["x = 1"] * 40
        occ = [(i, 0, 3, "local 0") for i in range(40)]  # 'x =' is not an identifier
        assert sc.fresh_occurrence_map(lines, occ) is None

    def test_dotted_import_paths_and_few_quirks_are_not_stale(self):
        lines = ["import rich.console", "x = f'{config.key}'"] + ["y = z"] * 20
        occ = [(0, 7, 19, "rich.console")] + [(1, 9, 15, "local 0")]  # f-string offset quirk
        occ += [(i, 0, 1, "local 1") for i in range(2, 22)]
        assert sc.fresh_occurrence_map(lines, occ) is not None

    def test_shifted_range_is_ignored_per_token(self):
        """A lookup whose range length differs from the name falls back to the name."""
        src = "def f(k):\n    v = helper(k)\n    return v + v * 2 + k\n"
        node = ast.parse(src).body[0]
        wrong = {(2, 8): (3, PKG + "`m`/other().")}  # 'helper' is 6 long
        right = {(2, 8): (6, PKG + "`m`/helper().")}
        assert "G:helper" in sc.normalize_function(node, wrong)
        assert "G:`m`/helper()." in sc.normalize_function(node, right)

    def test_load_scip_occurrences_reads_single_line_ranges(self, tmp_path: Path):
        scip_pb2 = pytest.importorskip("victor.core.graph_rag._scip.scip_pb2")
        index = scip_pb2.Index()
        doc = index.documents.add(relative_path="pkg/m.py")
        doc.occurrences.add(range=[1, 11, 17], symbol=PKG + "`m`/helper().")
        doc.occurrences.add(range=[1, 0, 2, 5], symbol="local 1")  # multi-line: skipped
        dump = tmp_path / "scip-python.scip"
        dump.write_bytes(index.SerializeToString())
        assert sc.load_scip_occurrences(dump) == {"pkg/m.py": [(1, 11, 17, PKG + "`m`/helper().")]}
        assert sc.load_scip_occurrences(tmp_path / "missing.scip") is None


class TestScan:
    def _repo(self, tmp_path: Path) -> Path:
        files = {
            "src/a.py": BODY_A,
            "src/b.py": BODY_A_RENAMED,
            "src/c.py": BODY_A_OTHER_CALLEE,
            "src/d.py": BODY_A_EDITED,
            "tests/test_a.py": BODY_A,
        }
        for rel, body in files.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(body))
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        return tmp_path

    def test_scan_reports_tiers_and_falls_back_without_scip(self, tmp_path: Path):
        root = self._repo(tmp_path)
        report = sc.scan(root, min_nodes=10, near_threshold=0.5)
        tiers = {f.tier: f.files for f in report.families}
        assert tiers["exact"] == ["src/a.py", "src/b.py"]
        assert set(tiers["template"]) >= {"src/a.py", "src/c.py"}
        assert "src/d.py" in tiers["near"]
        assert report.stats["scip_dump"] is False
        assert report.stats["ast_only_files"] == 4  # tests/ excluded by default
        assert not any("tests/" in f for fam in report.families for f in fam.files)

    def test_scope_and_include_tests(self, tmp_path: Path):
        root = self._repo(tmp_path)
        report = sc.scan(root, scope="tests/", include_tests=True, min_nodes=10)
        assert report.stats["functions"] == 1 and report.families == []
