"""Semantic near-duplicate discovery over the indexed symbol embeddings.

Reads the LanceDB embedding store produced by ``victor graph index
--embeddings``, computes cross-file nearest neighbours per symbol, and
groups them into similarity families — catching *evolved* copies that
exact-token tools (jscpd) cannot see. Read-only analysis; surfaced as
``victor graph similar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_THRESHOLD = 0.90


@dataclass(frozen=True)
class SimilarPair:
    left_idx: int
    right_idx: int
    similarity: float


@dataclass
class SymbolRecord:
    id: str
    file_path: str
    symbol_name: str
    kind: str
    line_number: int


@dataclass
class CloneFamily:
    similarity: float
    members: List[SymbolRecord] = field(default_factory=list)

    @property
    def files(self) -> List[str]:
        return sorted({m.file_path for m in self.members})


TEST_PATH_MARKERS = ("/tests/", "/test_", "/__tests__/", "/goldens/")


def _in_scope(file_path: str, scope: Optional[str], include_tests: bool) -> bool:
    if scope and not file_path.startswith(scope):
        return False
    if not include_tests and any(m in file_path for m in TEST_PATH_MARKERS):
        return False
    return True


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _scope_predicate(scope: Optional[str], include_tests: bool) -> str:
    """SQL WHERE for the store query: degenerate-name + scope + test filters.

    Pushed into LanceDB — a whole-repo store holds ~1M rows and
    materializing them client-side to filter costs minutes and gigabytes
    the pushdown avoids.
    """
    conds = ["symbol_name NOT LIKE '%:%'"]  # degenerate statement names
    if scope:
        conds.append(f"file_path LIKE '{_esc(scope)}%'")
    if not include_tests:
        for marker in TEST_PATH_MARKERS:
            conds.append(f"file_path NOT LIKE '%{_esc(marker)}%'")
    return " AND ".join(conds)


def load_embedding_symbols(
    project_root,
    scope: Optional[str] = None,
    include_tests: bool = False,
) -> Tuple[List[SymbolRecord], Any]:
    """Load (symbol records, vector matrix) from the project's LanceDB store.

    ``scope`` filters by file-path prefix; test trees are excluded unless
    ``include_tests`` — a whole-repo store embeds every symbol, and the
    duplicate scan wants source trees. Raises FileNotFoundError when the
    store is absent or empty — the caller turns that into the reindex hint.
    """
    import os

    import lancedb
    import numpy as np

    store_dir = os.path.join(str(project_root), ".victor", "embeddings")
    if not os.path.isdir(store_dir):
        raise FileNotFoundError(store_dir)
    table = lancedb.connect(store_dir).open_table("embeddings")

    # Push scope/degenerate filtering into the store query — a whole-repo
    # store holds ~1M rows and materializing them client-side to filter
    # costs minutes and gigabytes the pushdown avoids.
    rows = table.search().where(_scope_predicate(scope, include_tests)).limit(2_000_000).to_list()
    if not rows:
        raise FileNotFoundError(store_dir)

    # Safety net: the pushdown covers the common shapes; keep the exact
    # client-side predicate for anything the SQL LIKEs miss.
    kept = [
        r
        for r in rows
        if _in_scope(r["file_path"], scope, include_tests) and ":" not in r["symbol_name"]
    ]
    if not kept:
        raise FileNotFoundError(store_dir)

    records = [
        SymbolRecord(
            id=r["id"],
            file_path=r["file_path"],
            symbol_name=r["symbol_name"],
            kind=r.get("embedding_kind", ""),
            line_number=r.get("line_number", 0),
        )
        for r in kept
    ]
    mat = np.array([r["vector"] for r in kept], dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return records, mat


BLOCK_ROWS = 2048
# At 0.90 this repo yields ~280K pairs; at 0.5, 3.2B (79% of all pairs).
MAX_PAIRS = 5_000_000


class TooManyPairsError(ValueError):
    """The threshold matched more pairs than ``max_pairs`` allows."""


def make_union_find(n: int):
    """Tiny path-halving union-find over ``range(n)``.

    Shared with ``victor.native.python.graph_algo`` — one implementation,
    two callers (the native copy chose the opposite union root; component
    results are identical either way, so the shared form is safe).
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def find_clone_families(
    records: List[SymbolRecord],
    mat: Any,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: Optional[int] = None,
    include_same_file: bool = False,
    progress: Optional[Any] = None,
    max_pairs: int = MAX_PAIRS,
) -> List[CloneFamily]:
    """Group symbols into cross-file semantic near-duplicate families.

    An exact threshold join: every pair with cosine similarity >=
    ``threshold`` (after the same-file filter) is an edge, and families are
    the connected components. Similarities are computed ``BLOCK_ROWS`` rows
    at a time against the not-yet-visited suffix (upper triangle), so a
    whole-repo store never materializes the ``n × n`` matrix and each pair
    is scored once. ``top_k`` optionally caps the neighbours kept per row;
    the default keeps them all. ``progress(done, total)`` is called per
    block when provided.

    Raises:
        TooManyPairsError: more than ``max_pairs`` pairs matched — checked
            per block, so a too-low threshold fails fast instead of
            exhausting memory.
    """
    import numpy as np

    n = len(records)
    if n == 0:
        return []

    file_ids: Dict[str, int] = {}
    fid = np.fromiter(
        (file_ids.setdefault(r.file_path, len(file_ids)) for r in records), dtype=np.int64, count=n
    )
    pair_i: List[Any] = []
    pair_j: List[Any] = []
    pair_s: List[Any] = []
    total_pairs = 0
    for b0 in range(0, n, BLOCK_ROWS):
        b1 = min(b0 + BLOCK_ROWS, n)
        if top_k is None:
            # upper triangle: columns from b0 on; inside the block's own
            # square only j > i survives, so each pair is scored once
            sims = mat[b0:b1] @ mat[b0:].T
            hit = sims >= threshold
            hit[:, : b1 - b0] = np.triu(hit[:, : b1 - b0], 1)
            col0 = b0
        else:
            sims = mat[b0:b1] @ mat.T
            sims[np.arange(b1 - b0), np.arange(b0, b1)] = -1.0
            if not include_same_file:
                sims[fid[b0:b1, None] == fid[None, :]] = -1.0
            k = min(top_k, n - 1)
            hit = np.zeros_like(sims, dtype=bool)
            np.put_along_axis(hit, np.argpartition(-sims, k - 1, axis=1)[:, :k], True, axis=1)
            hit &= sims >= threshold
            col0 = 0
        # count before extracting: at a low threshold one block alone can
        # hold 100M+ hits, and np.nonzero would allocate gigabytes for them
        total_pairs += int(np.count_nonzero(hit))
        if total_pairs > max_pairs:
            raise TooManyPairsError(
                f"threshold {threshold} matched over {max_pairs:,} pairs after "
                f"{b1:,}/{n:,} symbols; raise the threshold or cap neighbours"
            )
        rows, cols = np.nonzero(hit)
        i = rows + b0
        j = cols + col0
        if not include_same_file:
            cross = fid[i] != fid[j]
            rows, cols, i, j = rows[cross], cols[cross], i[cross], j[cross]
        pair_i.append(np.minimum(i, j))
        pair_j.append(np.maximum(i, j))
        pair_s.append(sims[rows, cols])
        if progress is not None:
            progress(b1, n)

    pi = np.concatenate(pair_i).tolist()
    pj = np.concatenate(pair_j).tolist()
    ps = np.concatenate(pair_s).tolist()

    find, union = make_union_find(n)
    for i, j in zip(pi, pj):
        union(i, j)

    best: Dict[int, float] = {}
    for i, sim in zip(pi, ps):
        root = find(i)
        if sim > best.get(root, -1.0):
            best[root] = sim

    groups: Dict[int, List[int]] = {}
    for i in sorted(set(pi) | set(pj)):
        groups.setdefault(find(i), []).append(i)

    families = [
        CloneFamily(
            similarity=best[root],
            members=sorted((records[m] for m in members), key=lambda r: r.file_path),
        )
        for root, members in groups.items()
        if include_same_file or len({records[m].file_path for m in members}) > 1
    ]
    families.sort(key=lambda f: f.similarity, reverse=True)
    return families
