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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Semantic near-duplicate family discovery (victor graph similar)."""

import numpy as np
import pytest

from victor.core.graph_rag.semantic_clones import (
    CloneFamily,
    SymbolRecord,
    find_clone_families,
)


def rec(i, file, name):
    return SymbolRecord(id=str(i), file_path=file, symbol_name=name, kind="semantic", line_number=1)


def vec(theta):
    v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    return v / np.linalg.norm(v)


class TestFindCloneFamilies:
    def test_cross_file_duplicates_group(self):
        records = [
            rec(0, "a.py", "run_check"),
            rec(1, "b.py", "run_check"),  # near-identical, other file
            rec(2, "c.py", "unrelated"),  # orthogonal
        ]
        mat = np.stack([vec(0.0), vec(0.001), vec(1.5)])
        families = find_clone_families(records, mat, threshold=0.99)
        assert len(families) == 1
        assert {m.file_path for m in families[0].members} == {"a.py", "b.py"}

    def test_same_file_pairs_excluded_by_default(self):
        records = [rec(0, "a.py", "one"), rec(1, "a.py", "two")]
        mat = np.stack([vec(0.0), vec(0.001)])
        assert find_clone_families(records, mat, threshold=0.99) == []
        fams = find_clone_families(records, mat, threshold=0.99, include_same_file=True)
        assert len(fams) == 1

    def test_threshold_gate(self):
        records = [rec(0, "a.py", "x"), rec(1, "b.py", "y")]
        mat = np.stack([vec(0.0), vec(0.4)])  # cos ~0.92
        assert find_clone_families(records, mat, threshold=0.99) == []
        assert len(find_clone_families(records, mat, threshold=0.90)) == 1

    def test_empty_and_single(self):
        assert find_clone_families([], np.zeros((0, 2), dtype=np.float32)) == []
        assert find_clone_families([rec(0, "a.py", "x")], np.stack([vec(0.0)])) == []

    def test_family_sorting_by_similarity(self):
        records = [
            rec(0, "a.py", "x"),
            rec(1, "b.py", "y"),
            rec(2, "c.py", "z"),
            rec(3, "d.py", "w"),
        ]
        mat = np.stack([vec(0.0), vec(0.01), vec(1.0), vec(1.005)])
        fams = find_clone_families(records, mat, threshold=0.9)
        assert len(fams) == 2
        assert fams[0].similarity >= fams[1].similarity


class TestScopePredicateAndHelpers:
    def test_predicate_default_shape(self):
        from victor.core.graph_rag.semantic_clones import TEST_PATH_MARKERS, _scope_predicate

        pred = _scope_predicate(None, include_tests=False)
        assert "symbol_name NOT LIKE '%:%'" in pred
        for marker in TEST_PATH_MARKERS:
            assert f"file_path NOT LIKE '%{marker}%'" in pred

    def test_predicate_scope_and_tests(self):
        from victor.core.graph_rag.semantic_clones import _scope_predicate

        pred = _scope_predicate("victor/agent/", include_tests=True)
        assert "file_path LIKE 'victor/agent/%'" in pred
        assert "NOT LIKE '%/tests/%'" not in pred

    def test_predicate_escapes_quotes(self):
        from victor.core.graph_rag.semantic_clones import _esc, _scope_predicate

        assert _esc("it's") == "it''s"
        assert "it''s" in _scope_predicate("it's/", True)

    def test_in_scope_rules(self):
        from victor.core.graph_rag.semantic_clones import _in_scope

        assert _in_scope("victor/core/x.py", None, False)
        # markers carry a leading slash — root-level test dirs (no parent
        # segment) are NOT excluded; pin that as the documented behavior
        assert _in_scope("tests/unit/x.py", None, False)
        assert not _in_scope("pkg/tests/unit/x.py", None, False)
        assert _in_scope("victor/agent/x.py", "victor/agent/", False)
        assert not _in_scope("victor/core/x.py", "victor/agent/", False)
        assert not _in_scope("pkg/tests/x.py", None, False)
        assert not _in_scope("pkg/goldens/x.py", None, False)

    def test_family_files_property(self):
        from victor.core.graph_rag.semantic_clones import CloneFamily, SymbolRecord

        fam = CloneFamily(
            similarity=1.0,
            members=[
                SymbolRecord("b", "z.py", "n", "semantic", 1),
                SymbolRecord("a", "a.py", "n", "semantic", 2),
                SymbolRecord("c", "a.py", "m", "semantic", 3),
            ],
        )
        assert fam.files == ["a.py", "z.py"]

    def test_load_missing_store_raises(self, tmp_path):
        from victor.core.graph_rag.semantic_clones import load_embedding_symbols

        with pytest.raises(FileNotFoundError):
            load_embedding_symbols(tmp_path)


def test_progress_callback_and_union_find():
    import numpy as np

    from victor.core.graph_rag.semantic_clones import (
        SymbolRecord,
        find_clone_families,
        make_union_find,
    )

    records = [
        SymbolRecord("0", "a.py", "s0", "semantic", 0),
        SymbolRecord("1", "b.py", "s1", "semantic", 1),
        SymbolRecord("2", "c.py", "s2", "semantic", 2),
        SymbolRecord("3", "d.py", "s3", "semantic", 3),
    ]
    base = np.zeros((4, 4), dtype=np.float32)
    base[[0, 1], 0] = 1.0  # pair A aligned
    base[[2, 3], 1] = 1.0  # pair B aligned
    calls = []
    fams = find_clone_families(
        records, base, threshold=0.9, progress=lambda d, t: calls.append((d, t))
    )
    assert len(fams) == 2
    assert calls and calls[-1][0] <= calls[-1][1]

    find, union = make_union_find(4)
    union(0, 1)
    assert find(0) == find(1) != find(2)


def _reference_families(records, mat, threshold):
    """Brute-force oracle: every cross-file pair >= threshold, union-find."""
    n = len(records)
    sims = mat @ mat.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold and records[i].file_path != records[j].file_path:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(records[i].id)
    return {
        frozenset(g)
        for g in groups.values()
        if len(g) > 1 and len({records[int(x)].file_path for x in g}) > 1
    }


class TestThresholdJoin:
    def test_cross_file_twin_found_when_nearest_neighbours_are_same_file(self):
        # E has three same-file siblings closer than its cross-file twin F;
        # F has three cross-file neighbours closer than E. A per-row top-3 cap
        # applied before the same-file filter drops the E-F pair entirely.
        rows = [
            ("x.py", "e", [1.0, 0.0, 0.0, 0.0]),
            ("x.py", "s1", [1.0, 0.01, 0.0, 0.0]),
            ("x.py", "s2", [1.0, 0.0, 0.01, 0.0]),
            ("x.py", "s3", [1.0, 0.0, 0.0, 0.01]),
            ("y.py", "f", [1.0, 0.0, 0.0, 0.3]),
            ("p.py", "g1", [1.0, 0.0, 0.0, 0.31]),
            ("q.py", "g2", [1.0, 0.0, 0.0, 0.32]),
            ("r.py", "g3", [1.0, 0.0, 0.0, 0.29]),
        ]
        records = [rec(i, f, n) for i, (f, n, _) in enumerate(rows)]
        mat = np.array([v for _, _, v in rows], dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        assert float(mat[0] @ mat[4]) >= 0.9  # E-F is a genuine pair

        families = find_clone_families(records, mat, threshold=0.9)
        names = {r.symbol_name for fam in families for r in fam.members}
        assert {"e", "f"} <= names

    def test_matches_brute_force_oracle_across_block_boundaries(self, monkeypatch):
        import victor.core.graph_rag.semantic_clones as sc

        monkeypatch.setattr(sc, "BLOCK_ROWS", 7)  # force many uneven blocks
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(12, 16)).astype(np.float32)
        mat = np.repeat(centers, 5, axis=0) + rng.normal(scale=0.08, size=(60, 16)).astype(
            np.float32
        )
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        records = [rec(i, f"f{i % 4}.py", f"s{i}") for i in range(60)]

        got = {
            frozenset(r.id for r in fam.members)
            for fam in sc.find_clone_families(records, mat, threshold=0.9)
        }
        assert got == _reference_families(records, mat, 0.9)

    def test_family_similarity_is_max_member_pair(self):
        records = [rec(0, "a.py", "x"), rec(1, "b.py", "x"), rec(2, "c.py", "x")]
        mat = np.stack([vec(0.0), vec(0.2), vec(0.4)])  # 0-1, 1-2 at cos .98
        families = find_clone_families(records, mat, threshold=0.95)
        assert len(families) == 1
        assert families[0].similarity == pytest.approx(float(np.cos(0.2)), abs=1e-6)

    def test_optional_neighbour_cap_limits_pairs(self):
        records = [rec(i, f"{i}.py", "same") for i in range(6)]
        mat = np.tile(vec(0.0), (6, 1))
        assert len(find_clone_families(records, mat, threshold=0.9)[0].members) == 6
        capped = find_clone_families(records, mat, threshold=0.9, top_k=1)
        # each row keeps one neighbour: still >= 2 members per family, never > 6
        assert all(2 <= len(f.members) <= 6 for f in capped)


class TestPairBudget:
    def _dense(self, n=40):
        records = [rec(i, f"{i}.py", "same") for i in range(n)]
        return records, np.tile(vec(0.0), (n, 1))  # every pair matches

    def test_low_threshold_fails_fast_instead_of_exhausting_memory(self, monkeypatch):
        import victor.core.graph_rag.semantic_clones as sc

        monkeypatch.setattr(sc, "BLOCK_ROWS", 8)
        records, mat = self._dense()
        seen = []
        # block 1 (rows 0-7) yields 39+...+32 = 284 pairs; block 2 pushes past 300
        with pytest.raises(sc.TooManyPairsError, match="raise the threshold"):
            sc.find_clone_families(
                records, mat, threshold=0.5, max_pairs=300, progress=lambda d, t: seen.append(d)
            )
        assert seen == [8]  # stopped in block 2 of 5, not after the full scan

    def test_within_budget_is_unaffected(self):
        from victor.core.graph_rag.semantic_clones import find_clone_families

        records, mat = self._dense()
        families = find_clone_families(records, mat, threshold=0.5, max_pairs=40 * 39 // 2)
        assert len(families) == 1 and len(families[0].members) == 40
