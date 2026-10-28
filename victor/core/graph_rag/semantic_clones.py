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
DEFAULT_TOP_K = 3


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
    # search().limit() without a query vector returns the full table scan; the
    # pyarrow path avoids the optional 'lance' dependency.
    rows = table.search().limit(2_000_000).to_list()
    if not rows:
        raise FileNotFoundError(store_dir)

    kept = [
        r
        for r in rows
        if _in_scope(r["file_path"], scope, include_tests)
        # Statement-level nodes embed with degenerate name-only text
        # ("<type>:<line>") — every one lands near-identical in vector space
        # and drowns the scan in 1.000-similarity junk families. Real symbol
        # names never contain ':'.
        and ":" not in r["symbol_name"]
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


BLOCK_ROWS = 4096


def find_clone_families(
    records: List[SymbolRecord],
    mat: Any,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    include_same_file: bool = False,
    progress: Optional[Any] = None,
) -> List[CloneFamily]:
    """Group symbols into cross-file semantic near-duplicate families.

    Block-chunked: similarities are computed ``BLOCK_ROWS × n`` at a time —
    a whole-repo store (100K+ symbols) never materializes the full
    ``n × n`` matrix. ``progress(done, total)`` is called per block when
    provided.
    """
    import numpy as np

    n = len(records)
    if n == 0:
        return []

    pairs: Dict[Tuple[int, int], float] = {}
    for b0 in range(0, n, BLOCK_ROWS):
        block = mat[b0 : b0 + BLOCK_ROWS]
        sims = block @ mat.T  # (B, n)
        sims[np.arange(len(block)), np.arange(b0, b0 + len(block))] = -1.0
        top = np.argpartition(-sims, min(top_k, n - 1), axis=1)[:, :top_k]
        for bi in range(len(block)):
            i = b0 + bi
            for j in top[bi]:
                j = int(j)
                sim = float(sims[bi, j])
                if sim < threshold:
                    continue
                if records[i].file_path == records[j].file_path and not include_same_file:
                    continue
                key = (min(i, j), max(i, j))
                pairs[key] = max(pairs.get(key, 0.0), sim)
        if progress is not None:
            progress(min(b0 + BLOCK_ROWS, n), n)

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

    for i, j in pairs:
        union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    families = []
    for members in groups.values():
        if len(members) < 2:
            continue
        if not include_same_file and len({records[m].file_path for m in members}) < 2:
            continue
        sim = max(
            (pairs.get((min(a, b), max(a, b)), 0.0) for a in members for b in members if a != b),
            default=0.0,
        )
        families.append(
            CloneFamily(
                similarity=sim,
                members=sorted((records[m] for m in members), key=lambda r: r.file_path),
            )
        )
    families.sort(key=lambda f: f.similarity, reverse=True)
    return families
